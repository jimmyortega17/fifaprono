"""
Núcleo común: carga de datos del repositorio, matriz de marcadores Dixon-Coles, Elo y métricas.

Todo modelo del proyecto nuevo devuelve, para cada partido, una MATRIZ M[i, j] = P(local i goles, visita j goles).
De esa matriz salen el 1X2, el marcador más probable y cualquier mercado.
"""
from collections import namedtuple
from dataclasses import dataclass
from math import lgamma

import numpy as np

MAXG = 10
_LOGFACT = np.array([lgamma(k + 1) for k in range(MAXG + 1)])

# t = índice temporal (ronda); hadv = +1 si el equipo 1 juega en casa, -1 si el equipo 2, 0 si cancha neutral
Partido = namedtuple('Partido', 't h a gh ga hadv ronda')


@dataclass
class Liga:
    nombre: str
    equipos: list
    partidos: list
    elo0: np.ndarray        # Elo previo por equipo
    k_elo: float
    hfa_elo: float          # ventaja de casa en puntos Elo

    @property
    def n(self):
        return len(self.equipos)

    def idx(self, nombre):
        return self.equipos.index(nombre)

    def rondas(self):
        return sorted({p.t for p in self.partidos})

    def etiqueta(self, t):
        return next(p.ronda for p in self.partidos if p.t == t)


def cargar_liga1():
    from datos_liga1_2026 import PARTIDOS
    equipos = sorted({p[2] for p in PARTIDOS} | {p[3] for p in PARTIDOS})
    ix = {e: i for i, e in enumerate(equipos)}
    ps = [Partido(f if tor == 'A' else 17 + f, ix[h], ix[a], gh, ga, 1, f'{tor}{f}')
          for tor, f, h, a, gh, ga in PARTIDOS]
    ps.sort(key=lambda p: p.t)
    return Liga('Liga 1 Perú 2026', equipos, ps, np.full(len(equipos), 1500.0), 20.0, 60.0)


def cargar_mundial():
    import datos_mundial2026 as D
    equipos = sorted(D.ELO0)
    ix = {e: i for i, e in enumerate(equipos)}
    ps = []
    for t, (nombre, ronda) in enumerate(D.RONDAS, start=1):
        for h, gh, ga, a in ronda:
            hadv = 1 if h in D.ANFITRIONES else (-1 if a in D.ANFITRIONES else 0)
            ps.append(Partido(t, ix[h], ix[a], gh, ga, hadv, nombre))
    return Liga('Mundial 2026', equipos, ps, np.array([float(D.ELO0[e]) for e in equipos]), 40.0, 35.0)


# ---------------------------------------------------------------- Dixon-Coles
def pois_pmf(lam):
    k = np.arange(MAXG + 1)
    return np.exp(-lam + k * np.log(max(lam, 1e-9)) - _LOGFACT)


def dc_matrix(lh, la, rho):
    M = np.outer(pois_pmf(lh), pois_pmf(la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 1e-12, None)
    return M / M.sum()


def probs_1x2(M):
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def marcador_modal(M):
    i, j = np.unravel_index(M.argmax(), M.shape)
    return int(i), int(j)


def top_marcadores(M, k=3):
    orden = np.argsort(M, axis=None)[::-1][:k]
    return [(int(i), int(j), float(M[i, j])) for i, j in (np.unravel_index(o, M.shape) for o in orden)]


def goles_esperados(M):
    g = np.arange(M.shape[0])
    return float((M.sum(1) * g).sum()), float((M.sum(0) * g).sum())


# ------------------------------------------------------------------------ Elo
def elo_delta(liga, Rh, Ra, hadv, gh, ga):
    E = 1 / (1 + 10 ** (-(Rh + liga.hfa_elo * hadv - Ra) / 400))
    S = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
    gd = abs(gh - ga)
    mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
    return liga.k_elo * mult * (S - E)


def elo_recorrido(liga, train):
    """Devuelve (ratings finales, diferencia Elo previa a cada partido de train en unidades de 400)."""
    R = liga.elo0.copy()
    dE = np.zeros(len(train))
    for i, p in enumerate(train):
        dE[i] = (R[p.h] - R[p.a]) / 400
        d = elo_delta(liga, R[p.h], R[p.a], p.hadv, p.gh, p.ga)
        R[p.h] += d
        R[p.a] -= d
    return R, dE


# -------------------------------------------------------------------- métricas
def res_idx(gh, ga):
    return 0 if gh > ga else (1 if gh == ga else 2)


def metricas(mats, reales, picks=None):
    """mats: lista de matrices; reales: lista de (gh, ga); picks: marcadores elegidos (por defecto, el modo)."""
    n = len(mats)
    P = np.array([probs_1x2(M) for M in mats])
    y = np.array([res_idx(gh, ga) for gh, ga in reales])
    Y = np.eye(3)[y]
    picks = picks or [marcador_modal(M) for M in mats]
    exacto = np.array([tuple(pk) == (gh, ga) for pk, (gh, ga) in zip(picks, reales)])
    top3 = np.array([(gh, ga) in [(i, j) for i, j, _ in top_marcadores(M, 3)] for M, (gh, ga) in zip(mats, reales)])
    ll_marc = np.array([-np.log(M[min(gh, MAXG), min(ga, MAXG)]) for M, (gh, ga) in zip(mats, reales)])
    eg = np.array([goles_esperados(M) for M in mats])
    R = np.array(reales, float)
    return dict(
        n=n,
        acc=float((P.argmax(1) == y).mean()),
        exacto=float(exacto.mean()), n_exacto=int(exacto.sum()),
        top3=float(top3.mean()),
        logloss=float(-np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).mean()),
        brier=float(((P - Y) ** 2).sum(1).mean()),
        rps=float(((np.cumsum(P, 1)[:, :2] - np.cumsum(Y, 1)[:, :2]) ** 2).sum(1).mean() / 2),
        ll_marcador=float(ll_marc.mean()),
        mae_goles=float(np.abs(eg - R).mean()),
    )
