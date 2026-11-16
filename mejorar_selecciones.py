"""
MEJORA VALIDADA del pronóstico de selecciones. Tres pasos, cada uno probado fuera de muestra:

  A) Hiperparámetros del modelo estadístico y peso del ensamble: se eligen con partidos de
     ene-2026 al 10-jun-2026 (antes del periodo de prueba) y se comprueban del 11-jun en adelante.
  B) Corrección de goles local/visita: se ajusta con el periodo de elección y se comprueba en el de prueba.
  C) Factor de goles de la ÚLTIMA JORNADA de la Nations League: goles previstos por el modelo estadístico
     contra goles reales en las últimas jornadas de 2018, 2020, 2022 y 2024.

  python mejorar_selecciones.py
"""
import csv
from datetime import date
from itertools import product

import numpy as np

import modelo_deep as D
import modelo_mejorado as mm
import pronosticar_selecciones as S
from nucleo import MAXG, Partido, goles_esperados, probs_1x2
from preentrenar import CORTE, CSV

XIS, RIDGES, ALPHAS = (0.0005, 0.001, 0.002), (4.0, 8.0, 16.0), (0.25, 0.5, 0.75, 1.0)
I, J = np.indices((MAXG + 1, MAXG + 1))


def recorrer(filas, dias, con_red=True, solo=None):
    """Para cada día: matrices del estadístico por configuración y de la red. Devuelve lista de registros."""
    reg = []
    for d in dias:
        ix, l_deep, l_est = S.preparar(filas, d)
        hoy = [f for f in filas if f[0] == d and f[1] in ix and f[2] in ix and (solo is None or (f[0], f[1], f[2]) in solo)]
        if not hoy:
            continue
        t = d.toordinal()
        test = [Partido(t, ix[f[1]], ix[f[2]], 0, 0, f[5], '') for f in hoy]
        est = {}
        for xi, r in product(XIS, RIDGES):
            comp = mm.Componentes(l_est, l_est.partidos, t, xi, r, 1e4)
            for a in ALPHAS:
                est[(xi, r, a)] = comp.matrices(test, a)
        red = D.predecir(l_deep, l_deep.partidos, test, t, modo='cero') if con_red else [None] * len(hoy)
        for k, f in enumerate(hoy):
            reg.append(dict(gh=f[3], ga=f[4], est={c: v[k] for c, v in est.items()}, red=red[k]))
    return reg


def ll(M, r):
    return -np.log(M[min(r['gh'], MAXG), min(r['ga'], MAXG)])


def resumen(Ms, reg):
    P = np.array([probs_1x2(M) for M in Ms]); y = np.array([0 if r['gh'] > r['ga'] else 1 if r['gh'] == r['ga'] else 2 for r in reg])
    eg = np.array([goles_esperados(M) for M in Ms])
    return (f"ganador {np.mean(P.argmax(1) == y):.1%} · logloss 1X2 {-np.log(P[np.arange(len(y)), y]).mean():.4f} · "
            f"logloss marcador {np.mean([ll(M, r) for M, r in zip(Ms, reg)]):.4f} · goles previstos {eg[:, 0].mean():.2f}-{eg[:, 1].mean():.2f} "
            f"(reales {np.mean([r['gh'] for r in reg]):.2f}-{np.mean([r['ga'] for r in reg]):.2f})")


def inclinar(M, a, b):
    W = M * np.exp(a * I + b * J)
    return W / W.sum()


def main():
    from scipy.optimize import minimize
    filas = S.leer(date(2100, 1, 1))
    d_val = sorted({f[0] for f in filas if date(2026, 1, 1) <= f[0] < CORTE})
    d_pru = sorted({f[0] for f in filas if f[0] >= CORTE})
    val, pru = recorrer(filas, d_val), recorrer(filas, d_pru)
    print(f'Periodo de elección: {len(val)} partidos · periodo de prueba: {len(pru)} partidos\n')

    # ---- A) hiperparámetros y peso
    mejor = min(val[0]['est'], key=lambda c: np.mean([ll(r['est'][c], r) for r in val]))
    actual = (0.001, 8.0, 0.5)
    w_mejor = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda w: np.mean([ll(w * r['est'][mejor] + (1 - w) * r['red'], r) for r in val]))
    print(f'A) configuración elegida (xi, ridge, alpha) = {mejor}; peso del estadístico en el ensamble = {w_mejor}')
    ens = lambda r, c, w: w * r['est'][c] + (1 - w) * r['red']
    print('   PRUEBA, versión anterior :', resumen([ens(r, actual, 0.5) for r in pru], pru))
    print('   PRUEBA, versión elegida  :', resumen([ens(r, mejor, w_mejor) for r in pru], pru))

    # ---- B) corrección local/visita ajustada en el periodo de elección
    base_val = [ens(r, mejor, w_mejor) for r in val]
    ab = minimize(lambda p: sum(ll(inclinar(M, *p), r) for M, r in zip(base_val, val)), [0, 0], method='Nelder-Mead').x
    print(f'\nB) corrección de goles ajustada en elección: local x{np.exp(ab[0]):.3f}, visita x{np.exp(ab[1]):.3f}')
    print('   PRUEBA, con corrección   :', resumen([inclinar(ens(r, mejor, w_mejor), *ab) for r in pru], pru))

    # ---- C) última jornada de la Nations League en ediciones anteriores (solo modelo estadístico)
    nl, solo = {}, set()
    for r in csv.DictReader(open(CSV, encoding='utf-8')):
        if r['tournament'] == 'UEFA Nations League':
            d = date.fromisoformat(r['date'])
            if d.month >= 9 and d.year in (2018, 2020, 2022, 2024):
                nl.setdefault(d.year, set()).add(d)
                solo.add((d, r['home_team'], r['away_team']))
    print('\nC) última jornada de la Nations League, goles previstos por el modelo vs reales')
    tot_p = tot_r = n = 0
    prev_p = prev_r = m = 0
    for anio, dias in sorted(nl.items()):
        dias = sorted(dias)
        fin = [d for d in dias if (dias[-1] - d).days <= 2]
        antes = [d for d in dias if (dias[-1] - d).days > 2]
        for grupo, nombre in ((antes, 'previas'), (fin, 'última')):
            reg = recorrer(filas, grupo, con_red=False, solo=solo)
            # solo partidos de Nations League de esos días
            p = np.mean([sum(goles_esperados(r['est'][mejor])) for r in reg]); real = np.mean([r['gh'] + r['ga'] for r in reg])
            print(f'   {anio} {nombre:<8} {len(reg):>3} partidos: previsto {p:.2f}, real {real:.2f}, razón {real / p:.2f}')
            if nombre == 'última':
                tot_p += p * len(reg); tot_r += real * len(reg); n += len(reg)
            else:
                prev_p += p * len(reg); prev_r += real * len(reg); m += len(reg)
    print(f'   TOTAL jornadas previas ({m}): razón real/previsto {prev_r / prev_p:.3f}')
    print(f'   TOTAL última jornada  ({n}): razón real/previsto {tot_r / tot_p:.3f}')


if __name__ == '__main__':
    main()
