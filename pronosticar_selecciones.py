"""
PRONÓSTICO DE PARTIDOS DE SELECCIONES (p. ej. UEFA Nations League) con el ensamble 50/50:
  - red neuronal preentrenada (modelo_deep, modo 'cero') sobre el año en curso, y
  - modelo estadístico mejorado ajustado con los partidos internacionales desde 2024-07.

Datos: datos_externos/international_results/results.csv (se descarga con preentrenar.py; actualízalo con git pull).
Los nombres de equipos van en inglés, como en ese CSV. Un '*' tras el local indica cancha neutral.

  python pronosticar_selecciones.py                      # jornada del 15 al 17-nov-2026
  python pronosticar_selecciones.py --partidos "Spain-England;Belarus*-Albania"
  python pronosticar_selecciones.py --validar            # prueba fuera de muestra desde el 11-jun-2026
"""
import argparse
from datetime import date

import numpy as np

import modelo_deep as D
import modelo_mejorado as mm
from nucleo import Liga, Partido, elo_delta, goles_esperados, metricas, probs_1x2, top_marcadores
from preentrenar import CORTE, HFA, K_ELO, leer

JORNADA = """Cyprus-Montenegro;Latvia-Armenia;Finland-San Marino;Belarus*-Albania;France-Turkey;Belgium-Italy;
Croatia-Czech Republic;Spain-England;Faroe Islands-Moldova;Kazakhstan-Slovakia;Iceland-Luxembourg;
Lithuania-Liechtenstein;Estonia-Bulgaria;Germany-Netherlands;Switzerland-Scotland;Greece-Serbia;
North Macedonia-Slovenia;Gibraltar-Malta;Denmark-Norway;Wales-Portugal;Ukraine*-Georgia;Hungary-Northern Ireland;
Israel*-Austria;Republic of Ireland-Kosovo;Bosnia and Herzegovina-Romania;Poland-Sweden"""
HP_SEL = mm.HP(xi=0.001, ridge=8.0, ridge_h=1e4, alpha=0.5)     # xi por día (≈0.03 por mes)


def preparar(filas, hasta):
    """Ligas para predecir partidos del día `hasta`: (deep: año en curso; estadístico: desde jul-2024)."""
    filas = [f for f in filas if f[0] < hasta]
    equipos = sorted({f[1] for f in filas} | {f[2] for f in filas})
    ix = {e: i for i, e in enumerate(equipos)}
    ref = Liga('ref', equipos, [], np.full(len(equipos), 1500.0), K_ELO, HFA)
    R = ref.elo0.copy()
    ini_deep, ini_est = date(hasta.year, 1, 1), date(hasta.year - 2, 7, 1)
    R_deep = R_est = None
    ps = []
    for f in filas:
        if R_est is None and f[0] >= ini_est:
            R_est = R.copy()
        if R_deep is None and f[0] >= ini_deep:
            R_deep = R.copy()
        p = Partido(f[0].toordinal(), ix[f[1]], ix[f[2]], f[3], f[4], f[5], '')
        ps.append(p)
        d = elo_delta(ref, R[p.h], R[p.a], p.hadv, p.gh, p.ga)
        R[p.h] += d; R[p.a] -= d
    R_deep = R.copy() if R_deep is None else R_deep
    l_deep = Liga('deep', equipos, [p for p in ps if p.t >= ini_deep.toordinal()], R_deep, K_ELO, HFA)
    l_est = Liga('est', equipos, [p for p in ps if p.t >= ini_est.toordinal()], R_est, K_ELO, HFA)
    return ix, l_deep, l_est


def predecir(filas, hasta, cruces):
    """cruces: lista de (local, visita, hadv). Devuelve matrices (estadístico, deep, ensamble)."""
    ix, l_deep, l_est = preparar(filas, hasta)
    t = hasta.toordinal()
    test = [Partido(t, ix[h], ix[a], 0, 0, hadv, '') for h, a, hadv in cruces]
    est = mm.predecir(l_est, l_est.partidos, test, t, HP_SEL)
    deep = D.predecir(l_deep, l_deep.partidos, test, t, modo='cero')
    return est, deep, [0.5 * e + 0.5 * d for e, d in zip(est, deep)]


def validar(filas):
    """Walk-forward por día desde el corte del preentrenamiento: todo lo evaluado es fuera de muestra."""
    dias = sorted({f[0] for f in filas if f[0] >= CORTE})
    acum = {'Estadístico': [], 'Red neuronal': [], 'Ensamble 50/50': []}
    reales = []
    for d in dias:
        hoy = [f for f in filas if f[0] == d]
        e, n, m = predecir(filas, d, [(f[1], f[2], f[5]) for f in hoy])
        acum['Estadístico'] += e; acum['Red neuronal'] += n; acum['Ensamble 50/50'] += m
        reales += [(f[3], f[4]) for f in hoy]
    print(f'Validación fuera de muestra: {len(reales)} partidos internacionales del {dias[0]} al {dias[-1]}')
    print(f"{'Modelo':<16}{'1X2':>7}{'Exacto':>9}{'Top-3':>8}{'LogLoss':>9}{'RPS':>7}")
    for k, v in acum.items():
        m = metricas(v, reales)
        print(f"{k:<16}{m['acc']:>7.1%}{m['exacto']:>9.1%}{m['top3']:>8.1%}{m['logloss']:>9.3f}{m['rps']:>7.3f}")
    # mercado "más/menos de 2.5" del ensamble
    i, j = np.indices(acum['Ensamble 50/50'][0].shape)
    po = np.array([M[i + j > 2.5].sum() for M in acum['Ensamble 50/50']]); yo = np.array([gh + ga > 2.5 for gh, ga in reales])
    print(f'Más/menos de 2.5 goles (ensamble): acierto {((po > 0.5) == yo).mean():.1%}; frecuencia real de más de 2.5: {yo.mean():.1%}')
    P = np.array([probs_1x2(M) for M in acum['Ensamble 50/50']]); y = np.array([0 if a > b else 1 if a == b else 2 for a, b in reales])
    for lo, hi in ((0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)):
        s = (P.max(1) >= lo) & (P.max(1) < hi)
        if s.any():
            print(f'  favorito con {lo:.0%}-{min(hi, 1):.0%}: {s.sum():>3} partidos, acierto real {(P.argmax(1)[s] == y[s]).mean():.0%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partidos')
    ap.add_argument('--validar', action='store_true')
    a = ap.parse_args()
    filas = leer(date(2100, 1, 1))
    if a.validar:
        return validar(filas)
    cruces = []
    for c in (a.partidos or JORNADA).replace('\n', '').split(';'):
        h, v = (x.strip() for x in c.split('-', 1))
        cruces.append((h.rstrip('*'), v, 0 if h.endswith('*') else 1))
    hoy = date.today()
    est, deep, ens = predecir(filas, hoy, cruces)
    print(f'Datos hasta {filas[-1][0]} · pronóstico con fecha {hoy}\n')
    print(f"{'Partido':<36}{'L':>5}{'E':>5}{'V':>5}  {'Marcador':<9}{'Goles':>6}{'+2.5':>6}   Estad. L/E/V   Red L/E/V")
    i, j = np.indices(ens[0].shape)
    for (h, v, hadv), e, n, M in zip(cruces, est, deep, ens):
        p, pe, pn = probs_1x2(M), probs_1x2(e), probs_1x2(n)
        x, y, q = top_marcadores(M, 1)[0]
        lh, la = goles_esperados(M)
        nom = f"{h}{' (neutral)' if hadv == 0 else ''} - {v}"
        print(f"{nom:<36}{p[0]:>5.0%}{p[1]:>5.0%}{p[2]:>5.0%}  {f'{x}-{y} ({q:.0%})':<9}{lh + la:>6.2f}{M[i + j > 2.5].sum():>6.0%}"
              f"   {pe[0]:.0%}/{pe[1]:.0%}/{pe[2]:.0%}   {pn[0]:.0%}/{pn[1]:.0%}/{pn[2]:.0%}")


if __name__ == '__main__':
    main()
