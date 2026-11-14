"""
MODELO DE DEEP LEARNING (PyTorch) — red neuronal que predice la distribución completa del marcador.

Por qué está hecho en dos etapas
  Con los ~270 partidos de Liga 1 o los ~100 del Mundial una red se sobreajusta (se probó: ver README).
  Por eso la red se PREENTRENA con ~30 años de partidos internacionales (preentrenar.py, datos públicos
  CC0 anteriores al 11-jun-2026, o sea sin ningún partido del backtest) y luego se AFINA con cada liga.

Arquitectura
  entradas : variables pre-partido (ΔElo dinámico, ΔElo previo, casa/visita/neutral, goles a favor y en
             contra por partido, forma de los últimos 5, rendimiento en esa condición, partidos jugados).
  núcleo   : MLP residual (GELU + dropout) sobre un prior Elo; aprende cómo se traducen fuerza, forma y
             localía en goles. Se entrena con la log-verosimilitud del MARCADOR EXACTO (Poisson + Dixon-Coles).
  adaptador: capa por liga con efectos de equipo (ataque, defensa, casa) y sesgo, inicializada en cero.
  salida   : log λ_local, log λ_visita y ρ -> matriz de marcadores.

Modos de predicción (el backtest elige con las rondas pasadas):
  'cero'   : núcleo preentrenado tal cual (sin mirar la liga).
  'ajuste' : núcleo congelado + adaptador de equipos ajustado con la liga.
  'fino'   : además se afina el núcleo unos pocos pasos, atado a los pesos preentrenados.
Siempre se promedia un ensamble de varias redes (semillas distintas).
"""
import os

import numpy as np
import torch
import torch.nn as nn

from nucleo import MAXG, elo_delta

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

MU_PRIOR, C_PRIOR, G_PRIOR = float(np.log(1.3)), 0.5, 0.15
RUTA_PESOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pesos', 'preentrenado.pt')
_LOGFACT = torch.lgamma(torch.arange(MAXG + 1, dtype=torch.float64) + 1)


# ------------------------------------------------------------ variables pre-partido
class Estado:
    """Historia acumulada de cada equipo; solo contiene partidos ya jugados."""

    def __init__(self, liga):
        self.liga = liga
        self.R = liga.elo0.copy()
        self.hist = [[] for _ in range(liga.n)]   # (gf, ga, condición 'H'/'A'/'N', puntos)
        self.goles = 0.0
        self.njug = 0

    def _lado(self, e, cond):
        mu = (self.goles + 20 * 1.3) / (self.njug + 20)
        H = self.hist[e]
        n = len(H)
        gf = (sum(x[0] for x in H) + 3 * mu) / (n + 3)
        ga = (sum(x[1] for x in H) + 3 * mu) / (n + 3)
        u5 = H[-5:]
        forma = (sum(x[3] for x in u5) + 2 * 1.35) / (len(u5) + 2) / 3
        V = [x for x in H if x[2] == cond] if cond != 'N' else H
        vgf = (sum(x[0] for x in V) + 3 * gf) / (len(V) + 3)
        vga = (sum(x[1] for x in V) + 3 * ga) / (len(V) + 3)
        return [gf - mu, ga - mu, forma - 0.45, vgf - mu, vga - mu, np.log1p(n) / 3]

    def variables(self, h, a, hadv):
        ch, ca = ('H', 'A') if hadv > 0 else (('A', 'H') if hadv < 0 else ('N', 'N'))
        dE = (self.R[h] - self.R[a]) / 400
        d0 = (self.liga.elo0[h] - self.liga.elo0[a]) / 400
        return [dE, d0, float(hadv)] + self._lado(h, ch) + self._lado(a, ca)

    def actualizar(self, p):
        d = elo_delta(self.liga, self.R[p.h], self.R[p.a], p.hadv, p.gh, p.ga)
        self.R[p.h] += d
        self.R[p.a] -= d
        ch, ca = ('H', 'A') if p.hadv > 0 else (('A', 'H') if p.hadv < 0 else ('N', 'N'))
        ph = 3 if p.gh > p.ga else 1 if p.gh == p.ga else 0
        pa = 3 if p.ga > p.gh else 1 if p.gh == p.ga else 0
        self.hist[p.h].append((p.gh, p.ga, ch, ph))
        self.hist[p.a].append((p.ga, p.gh, ca, pa))
        self.goles += p.gh + p.ga
        self.njug += 2


NVAR = 15


def construir(liga, train, test=()):
    """Filas (x, h, a, hadv, gh, ga, t) de train (con su espejo) y de test, usando solo el pasado de cada partido."""
    est = Estado(liga)
    filas_tr = []
    train = sorted(train, key=lambda p: p.t)
    i = 0
    while i < len(train):
        j = i
        while j < len(train) and train[j].t == train[i].t:
            j += 1
        lote = train[i:j]
        for p in lote:   # las variables de toda la ronda se calculan antes de actualizar con sus resultados
            filas_tr.append((est.variables(p.h, p.a, p.hadv), p.h, p.a, p.hadv, p.gh, p.ga, p.t))
            filas_tr.append((est.variables(p.a, p.h, -p.hadv), p.a, p.h, -p.hadv, p.ga, p.gh, p.t))
        for p in lote:
            est.actualizar(p)
        i = j
    filas_te = [(est.variables(p.h, p.a, p.hadv), p.h, p.a, p.hadv, 0, 0, p.t) for p in test]
    return filas_tr, filas_te


def tensores(filas):
    X = torch.tensor([f[0] for f in filas]).reshape(-1, NVAR)
    L = lambda k: torch.tensor([f[k] for f in filas], dtype=torch.long)
    F = lambda k: torch.tensor([float(f[k]) for f in filas])
    return X, L(1), L(2), F(3), F(4), F(5), F(6)


# ------------------------------------------------------------------------- red
class Nucleo(nn.Module):
    def __init__(self, oculto=64, dropout=0.10):
        super().__init__()
        self.cuerpo = nn.Sequential(nn.Linear(NVAR, oculto), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(oculto, oculto // 2), nn.GELU(), nn.Dropout(dropout))
        self.cabeza = nn.Linear(oculto // 2, 2)
        nn.init.zeros_(self.cabeza.weight)
        nn.init.zeros_(self.cabeza.bias)
        self.rho_crudo = nn.Parameter(torch.tensor(-0.25))
        self.register_buffer('mu', torch.zeros(NVAR))
        self.register_buffer('sd', torch.ones(NVAR))

    def forward(self, X, hadv):
        corr = self.cabeza(self.cuerpo((X - self.mu) / self.sd))
        base = C_PRIOR * (X[:, 0]) + G_PRIOR * hadv          # prior: ΔElo dinámico y ventaja de casa
        return MU_PRIOR + base + corr[:, 0], MU_PRIOR - base + corr[:, 1]

    @property
    def rho(self):
        return 0.2 * torch.tanh(self.rho_crudo)


class Adaptador(nn.Module):
    """Efectos por equipo de la liga concreta, sumados a la salida del núcleo."""

    def __init__(self, n):
        super().__init__()
        self.sesgo = nn.Parameter(torch.zeros(2))
        self.att = nn.Parameter(torch.zeros(n))
        self.de = nn.Parameter(torch.zeros(n))
        self.hh = nn.Parameter(torch.zeros(n))

    def forward(self, h, a, hadv):
        sH, sA = (hadv > 0).double(), (hadv < 0).double()
        casa = sH * self.hh[h] - sA * self.hh[a]
        b = self.sesgo[0] + self.sesgo[1] * hadv
        return b + casa + self.att[h] - self.de[a], self.sesgo[0] - self.sesgo[1] * hadv - casa + self.att[a] - self.de[h]

    def penal(self, ridge, ridge_h):
        return ridge * ((self.att ** 2).sum() + (self.de ** 2).sum()) + ridge_h * (self.hh ** 2).sum() + 2.0 * (self.sesgo ** 2).sum()


def log_tau(gh, ga, lh, la, rho):
    uno = torch.ones_like(lh)
    tau = torch.where((gh == 0) & (ga == 0), 1 - lh * la * rho,
          torch.where((gh == 0) & (ga == 1), 1 + lh * rho,
          torch.where((gh == 1) & (ga == 0), 1 + la * rho,
          torch.where((gh == 1) & (ga == 1), 1 - rho * uno, uno))))
    return torch.log(tau.clamp_min(1e-6))


def log_ver(eh, ea, rho, gh, ga):
    lh, la = torch.exp(eh), torch.exp(ea)
    return gh * eh - lh - torch.lgamma(gh + 1) + ga * ea - la - torch.lgamma(ga + 1) + log_tau(gh, ga, lh, la, rho)


def matrices(eh, ea, rho):
    k = torch.arange(MAXG + 1, dtype=torch.float64)
    ph = torch.exp(-torch.exp(eh)[:, None] + k * eh[:, None] - _LOGFACT)
    pa = torch.exp(-torch.exp(ea)[:, None] + k * ea[:, None] - _LOGFACT)
    M = ph[:, :, None] * pa[:, None, :]
    lh, la = torch.exp(eh), torch.exp(ea)
    M[:, 0, 0] *= 1 - lh * la * rho
    M[:, 0, 1] *= 1 + lh * rho
    M[:, 1, 0] *= 1 + la * rho
    M[:, 1, 1] *= 1 - rho
    M = M.clamp_min(1e-12)
    return M / M.sum((1, 2), keepdim=True)


# ------------------------------------------------------------------ predicción
_CACHE = {}


def cargar_nucleos(ruta=RUTA_PESOS):
    if ruta not in _CACHE:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f'Falta {ruta}. Ejecuta primero:  python preentrenar.py')
        paquete = torch.load(ruta, weights_only=True)
        redes = []
        for sd in paquete['estados']:
            r = Nucleo(paquete['oculto'], paquete['dropout'])
            r.load_state_dict(sd)
            r.eval()
            redes.append(r)
        _CACHE[ruta] = redes
    return _CACHE[ruta]


def predecir(liga, train, test, t_ahora, modo='ajuste', ridge=8.0, ridge_h=30.0, xi=0.02,
             pasos_fino=40, lr_fino=5e-4, atadura=50.0):
    filas_tr, filas_te = construir(liga, train, test)
    Xte, hte, ate, hadvte, *_ = tensores(filas_te)
    usar_liga = modo != 'cero' and len(filas_tr) >= 16
    if usar_liga:
        Xtr, htr, atr, hadvtr, ghtr, gatr, ttr = tensores(filas_tr)
        w = 0.5 * torch.exp(-xi * (t_ahora - ttr))     # 0.5: cada partido aparece dos veces (espejo)
    acum = torch.zeros(len(test), MAXG + 1, MAXG + 1)
    nucleos = cargar_nucleos()
    for base in nucleos:
        red = Nucleo(base.cabeza.in_features * 2, 0.0)
        red.load_state_dict(base.state_dict())
        red.eval()
        ad = Adaptador(liga.n)
        if usar_liga:
            with torch.no_grad():
                eh0, ea0 = red(Xtr, hadvtr)
            rho = red.rho_crudo
            rho.requires_grad_(False)
            opt = torch.optim.LBFGS(ad.parameters(), lr=1.0, max_iter=100, line_search_fn='strong_wolfe')

            def cierre():
                opt.zero_grad()
                ch, ca = ad(htr, atr, hadvtr)
                v = -(w * log_ver(eh0 + ch, ea0 + ca, red.rho, ghtr, gatr)).sum() + ad.penal(ridge, ridge_h)
                v.backward()
                return v
            opt.step(cierre)
            if modo == 'fino':
                ref = [p.detach().clone() for p in red.parameters()]
                rho.requires_grad_(True)
                opt2 = torch.optim.Adam(list(red.parameters()) + list(ad.parameters()), lr=lr_fino)
                for _ in range(pasos_fino):
                    opt2.zero_grad()
                    eh, ea = red(Xtr, hadvtr)
                    ch, ca = ad(htr, atr, hadvtr)
                    v = -(w * log_ver(eh + ch, ea + ca, red.rho, ghtr, gatr)).sum() + ad.penal(ridge, ridge_h)
                    v = v + atadura * sum(((p - r) ** 2).sum() for p, r in zip(red.parameters(), ref))
                    v.backward()
                    opt2.step()
        with torch.no_grad():
            eh, ea = red(Xte, hadvte)
            ch, ca = ad(hte, ate, hadvte)
            acum += matrices(eh + ch, ea + ca, red.rho)
    return [m.numpy() for m in acum / len(nucleos)]
