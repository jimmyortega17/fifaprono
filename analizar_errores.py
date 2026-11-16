"""
ANÁLISIS DE ERRORES del pronóstico de selecciones, fuera de muestra (walk-forward por día desde el 11-jun-2026).
Revisa por mercado: ganador, doble oportunidad, marcador exacto, total de goles y ambos anotan,
y busca sesgos sistemáticos (goles de menos, exceso de confianza, etc.).

  python analizar_errores.py                 # todos los partidos y solo UEFA Nations League
"""
import csv
import pickle
import os
from datetime import date

import numpy as np

import pronosticar_selecciones as S
from nucleo import goles_esperados, probs_1x2, top_marcadores
from preentrenar import CORTE, CSV, EXTRA

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datos_externos', 'walkforward_selecciones.pkl')


def recolectar():
    filas = S.leer(date(2100, 1, 1))
    torneo = {}
    for ruta in (CSV, EXTRA):
        if os.path.exists(ruta):
            for r in csv.DictReader(open(ruta, encoding='utf-8')):
                torneo.setdefault((r['date'], r['home_team'], r['away_team']), r['tournament'])
    reg = []
    for d in sorted({f[0] for f in filas if f[0] >= CORTE}):
        hoy = [f for f in filas if f[0] == d]
        e, n, m = S.predecir(filas, d, [(f[1], f[2], f[5]) for f in hoy])
        for f, me, mn, mm_ in zip(hoy, e, n, m):
            reg.append(dict(fecha=d, h=f[1], a=f[2], gh=f[3], ga=f[4], hadv=f[5], torneo=torneo.get((str(d), f[1], f[2]), ''),
                            est=me, red=mn, ens=mm_))
    return reg


def calib(p, y, cortes):
    out = []
    for lo, hi in zip(cortes[:-1], cortes[1:]):
        s = (p >= lo) & (p < hi)
        if s.sum():
            out.append(f'    prob. {lo:.0%}-{min(hi, 1):.0%}: {int(s.sum()):>3} casos, previsto {p[s].mean():.0%}, real {y[s].mean():.0%}')
    return '\n'.join(out)


def informe(reg, titulo, clave='ens'):
    n = len(reg)
    Ms = [r[clave] for r in reg]
    gh = np.array([r['gh'] for r in reg]); ga = np.array([r['ga'] for r in reg]); tot = gh + ga
    i, j = np.indices(Ms[0].shape); t = i + j
    P = np.array([probs_1x2(M) for M in Ms]); y = np.where(gh > ga, 0, np.where(gh == ga, 1, 2))
    print(f'\n================ {titulo} — {n} partidos ================')
    # --- ganador
    pick = P.argmax(1)
    print(f'GANADOR: acierto {np.mean(pick == y):.1%}')
    for k, nom in enumerate(('local', 'empate', 'visita')):
        s = pick == k
        print(f'    pronosticó {nom:<7}: {int(s.sum()):>3} veces, acertó {np.mean(y[s] == k) if s.any() else 0:.0%}   | '
              f'frecuencia real de {nom}: {np.mean(y == k):.0%}, prob. media prevista {P[:, k].mean():.0%}')
    # --- doble oportunidad
    DC = np.stack([P[:, 0] + P[:, 1], P[:, 0] + P[:, 2], P[:, 1] + P[:, 2]], 1)
    Y = np.stack([y != 2, y != 1, y != 0], 1)
    k = DC.argmax(1); pm = DC[np.arange(n), k]; ok = Y[np.arange(n), k]
    print(f'DOBLE OPORTUNIDAD (la más probable): acierto {ok.mean():.1%}, previsto {pm.mean():.1%}')
    print(calib(pm, ok, [0.6, 0.7, 0.8, 0.9, 1.01]))
    for kk, nom in enumerate(('1X', '12', 'X2')):
        s = k == kk
        if s.any():
            print(f'    eligió {nom}: {int(s.sum()):>3} veces, acertó {ok[s].mean():.0%} (previsto {pm[s].mean():.0%})')
    fall = ~ok & (pm >= 0.8)
    print(f'    fallos con 80 % o más: {int(fall.sum())} de {int((pm >= 0.8).sum())}')
    # --- marcador
    top1 = np.array([top_marcadores(M, 1)[0][:2] == (a, b) for M, a, b in zip(Ms, gh, ga)])
    top3 = np.array([(a, b) in [x[:2] for x in top_marcadores(M, 3)] for M, a, b in zip(Ms, gh, ga)])
    top5 = np.array([(a, b) in [x[:2] for x in top_marcadores(M, 5)] for M, a, b in zip(Ms, gh, ga)])
    print(f'MARCADOR EXACTO: el más probable {top1.mean():.1%} · entre los 3 primeros {top3.mean():.1%} · entre los 5 primeros {top5.mean():.1%}')
    prev = np.array([sum(q for *_, q in top_marcadores(M, 3)) for M in Ms])
    print(f'    (los 3 primeros debían acertar {prev.mean():.1%} según el propio modelo)')
    de = sorted({(a, b) for a, b in zip(gh, ga)}, key=lambda s: -np.sum((gh == s[0]) & (ga == s[1])))[:8]
    print('    marcadores reales más frecuentes (real vs previsto): ' +
          ', '.join(f'{a}-{b} {np.mean((gh == a) & (ga == b)):.0%}/{np.mean([M[a, b] for M in Ms]):.0%}' for a, b in de))
    # --- goles
    eg = np.array([goles_esperados(M) for M in Ms])
    print(f'GOLES: media real {tot.mean():.2f} por partido, prevista {eg.sum(1).mean():.2f}  (local {gh.mean():.2f} vs {eg[:, 0].mean():.2f}; visita {ga.mean():.2f} vs {eg[:, 1].mean():.2f})')
    print('    total de goles, real vs previsto: ' + ', '.join(
        f'{g}: {np.mean(tot == g):.0%}/{np.mean([M[t == g].sum() for M in Ms]):.0%}' for g in range(6)) +
          f', 6+: {np.mean(tot >= 6):.0%}/{np.mean([M[t >= 6].sum() for M in Ms]):.0%}')
    for L in (1.5, 2.5, 3.5, 4.5):
        po = np.array([M[t > L].sum() for M in Ms]); yo = tot > L
        print(f'    más de {L}: real {yo.mean():.0%}, previsto {po.mean():.0%}; acierto apostando al lado más probable {np.mean((po > .5) == yo):.0%}')
    fav = P.max(1) >= 0.6
    print(f'    con favorito claro (60 %+, {int(fav.sum())} partidos): goles reales {tot[fav].mean():.2f} vs previstos {eg.sum(1)[fav].mean():.2f};'
          f' menos de 3.5 real {np.mean(tot[fav] < 3.5):.0%} vs previsto {np.mean([M[t < 3.5].sum() for M, f in zip(Ms, fav) if f]):.0%}')
    pb = np.array([M[(i > 0) & (j > 0)].sum() for M in Ms]); yb = (gh > 0) & (ga > 0)
    print(f'AMBOS ANOTAN: real {yb.mean():.0%}, previsto {pb.mean():.0%}; acierto {np.mean((pb > .5) == yb):.0%}')


def main():
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) > max(os.path.getmtime(CSV), os.path.getmtime(EXTRA)):
        reg = pickle.load(open(CACHE, 'rb'))
    else:
        reg = recolectar()
        pickle.dump(reg, open(CACHE, 'wb'))
    informe(reg, 'TODOS LOS PARTIDOS desde 11-jun-2026')
    nl = [r for r in reg if r['torneo'] == 'UEFA Nations League']
    informe(nl, 'UEFA NATIONS LEAGUE (sep-nov 2026)')
    informe(nl, 'UEFA NATIONS LEAGUE, solo modelo estadístico', 'est')
    informe(nl, 'UEFA NATIONS LEAGUE, solo red neuronal', 'red')
    informe([r for r in nl if r['fecha'] >= date(2026, 11, 12)], 'NATIONS LEAGUE, jornadas 5 y 6 (12 al 16-nov)')


if __name__ == '__main__':
    main()
