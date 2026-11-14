"""
MODELO ESTADÍSTICO MEJORADO — Poisson + Dixon-Coles con dos componentes combinadas:

  (A) Elo dinámico -> goles: log λ = m ± g·casa ± c·ΔElo  (regresión de Poisson con prior, funciona con 0 datos).
  (B) Ataque/defensa por equipo con contracción (ridge), decaimiento temporal, ancla en el Elo previo y
      VENTAJA DE CASA PROPIA DE CADA EQUIPO (clave en Liga 1 por la altura: Cusco, Huancayo, Juliaca...).

  Mezcla geométrica: log λ = (1-α)·log λ_A + α·log λ_B ; ρ de Dixon-Coles por máxima verosimilitud penalizada.

Mejoras respecto a modelo_liga1_walkforward.py / modelo_mundial2026_v2.py:
  - un solo código para Liga 1 y Mundial (casa, visita o cancha neutral);
  - ventaja de local por equipo; prior Elo dentro del ajuste de ataque/defensa;
  - hiperparámetros elegidos SOLO con rondas pasadas (ver backtest.py), no fijados a mano ni con el test;
  - sin constantes "mágicas" (boost de J3, corrección de goleada): el marcador sale del modo de la matriz.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from nucleo import dc_matrix, elo_recorrido

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class HP:
    xi: float = 0.02        # decaimiento temporal por ronda
    ridge: float = 8.0      # contracción de ataque/defensa
    ridge_h: float = 30.0   # contracción de la ventaja de casa propia de cada equipo
    alpha: float = 0.5      # peso de la componente ataque/defensa


HP_DEFECTO = HP()
REJILLA_BASE = [dict(xi=x, ridge=r, ridge_h=rh) for x, r, rh in product((0.0, 0.03), (4.0, 8.0, 16.0), (8.0, 30.0, 1e4))]
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def _lbfgs(params, perdida):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, tolerance_grad=1e-7, tolerance_change=1e-10,
                            line_search_fn='strong_wolfe')

    def cierre():
        opt.zero_grad()
        v = perdida()
        v.backward()
        return v
    opt.step(cierre)


def _arr(train):
    T = lambda v, dt=torch.float64: torch.tensor(v, dtype=dt)
    return dict(h=T([p.h for p in train], torch.long), a=T([p.a for p in train], torch.long),
                gh=T([float(p.gh) for p in train]), ga=T([float(p.ga) for p in train]),
                sH=T([1.0 if p.hadv > 0 else 0.0 for p in train]), sA=T([1.0 if p.hadv < 0 else 0.0 for p in train]),
                t=T([float(p.t) for p in train]))


class Componentes:
    """Ajusta (A) y (B) con los partidos de `train` y entrega los log-λ de cualquier partido."""

    def __init__(self, liga, train, t_ahora, xi, ridge, ridge_h):
        self.liga = liga
        train = sorted(train, key=lambda p: p.t)
        self.train = train
        d = _arr(train)
        self.R, dE = elo_recorrido(liga, train)
        dE = torch.tensor(dE)
        w = torch.exp(-xi * (t_ahora - d['t']))
        hadv = d['sH'] - d['sA']
        elo0 = torch.tensor(liga.elo0)
        d0 = (elo0[d['h']] - elo0[d['a']]) / 400 if train else torch.zeros(0)

        # ---- (A) mapa Elo -> goles, con prior suave
        pa = torch.tensor([np.log(1.3), 0.15, 0.5], requires_grad=True)

        def perd_a():
            m, g, c = pa
            eh = m + g * hadv + c * dE
            ea = m - g * hadv - c * dE
            ll = (w * (d['gh'] * eh - torch.exp(eh) + d['ga'] * ea - torch.exp(ea))).sum()
            prior = ((m - np.log(1.3)) / 0.5) ** 2 + ((g - 0.15) / 0.3) ** 2 + ((c - 0.5) / 0.5) ** 2
            return -ll + 0.5 * prior
        _lbfgs([pa], perd_a)
        self.pa = pa.detach().numpy()

        # ---- (B) ataque/defensa + casa por equipo + ancla Elo previo
        n = liga.n
        glob = torch.tensor([np.log(1.3), 0.15, 0.10, 0.5], requires_grad=True)   # mu, casa_ataque, casa_defensa, c0
        att = torch.zeros(n, requires_grad=True)
        de = torch.zeros(n, requires_grad=True)
        hh = torch.zeros(n, requires_grad=True)

        def perd_b():
            mu, g1, g2, c0 = glob
            eh = mu + d['sH'] * (g1 + hh[d['h']]) - d['sA'] * (g2 + hh[d['a']]) + c0 * d0 + att[d['h']] - de[d['a']]
            ea = mu + d['sA'] * (g1 + hh[d['a']]) - d['sH'] * (g2 + hh[d['h']]) - c0 * d0 + att[d['a']] - de[d['h']]
            ll = (w * (d['gh'] * eh - torch.exp(eh) + d['ga'] * ea - torch.exp(ea))).sum()
            pen = ridge * ((att ** 2).sum() + (de ** 2).sum()) + ridge_h * (hh ** 2).sum()
            prior = ((mu - np.log(1.3)) / 0.5) ** 2 + ((g1 - 0.15) / 0.3) ** 2 + ((g2 - 0.10) / 0.3) ** 2 + ((c0 - 0.5) / 0.5) ** 2
            return -ll + pen + 0.5 * prior
        _lbfgs([glob, att, de, hh], perd_b)
        self.glob, self.att, self.de, self.hh = (x.detach().numpy() for x in (glob, att, de, hh))

        self.train_ll = np.array([self.log_lambdas(p.h, p.a, p.hadv) for p in train]).reshape(-1, 4)

    def log_lambdas(self, h, a, hadv):
        """(log λh_A, log λa_A, log λh_B, log λa_B)."""
        m, g, c = self.pa
        dE = (self.R[h] - self.R[a]) / 400
        mu, g1, g2, c0 = self.glob
        sH, sA = float(hadv > 0), float(hadv < 0)
        d0 = (self.liga.elo0[h] - self.liga.elo0[a]) / 400
        bh = mu + sH * (g1 + self.hh[h]) - sA * (g2 + self.hh[a]) + c0 * d0 + self.att[h] - self.de[a]
        ba = mu + sA * (g1 + self.hh[a]) - sH * (g2 + self.hh[h]) - c0 * d0 + self.att[a] - self.de[h]
        return m + g * hadv + c * dE, m - g * hadv - c * dE, bh, ba

    def _rho(self, alpha):
        if not self.train:
            return -0.05
        L = self.train_ll
        lh = np.exp((1 - alpha) * L[:, 0] + alpha * L[:, 2])
        la = np.exp((1 - alpha) * L[:, 1] + alpha * L[:, 3])
        gh = np.array([p.gh for p in self.train]); ga = np.array([p.ga for p in self.train])
        m00, m01, m10, m11 = (gh == 0) & (ga == 0), (gh == 0) & (ga == 1), (gh == 1) & (ga == 0), (gh == 1) & (ga == 1)

        def nll(rho):
            tau = np.ones(len(gh))
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            if (tau <= 0).any():
                return 1e9
            return -np.log(tau).sum() + 0.5 * ((rho + 0.05) / 0.10) ** 2
        return float(minimize_scalar(nll, bounds=(-0.25, 0.20), method='bounded').x)

    def matrices(self, test, alpha):
        rho = self._rho(alpha)
        out = []
        for p in test:
            ea_h, ea_a, eb_h, eb_a = self.log_lambdas(p.h, p.a, p.hadv)
            out.append(dc_matrix(np.exp((1 - alpha) * ea_h + alpha * eb_h), np.exp((1 - alpha) * ea_a + alpha * eb_a), rho))
        return out


def predecir(liga, train, test, t_ahora, hp=HP_DEFECTO):
    return Componentes(liga, train, t_ahora, hp.xi, hp.ridge, hp.ridge_h).matrices(test, hp.alpha)
