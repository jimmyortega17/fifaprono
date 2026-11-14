"""
PRONÓSTICO de partidos aún no jugados con el ensamble (estadístico mejorado + red neuronal).

Usa la configuración que el backtest eligió con todas las rondas ya jugadas (seleccion.json).

  python pronosticar.py                                   # fecha 17 del Clausura de Liga 1
  python pronosticar.py --partidos "Melgar-Cusco FC;UTC-ADT"
  python pronosticar.py --liga mundial --partidos "ESP-ARG;FRA-ENG"
"""
import argparse
import json
import os

import numpy as np

import modelo_deep
import modelo_mejorado as mm
from backtest import VARIANTES_DEEP
from nucleo import Partido, cargar_liga1, cargar_mundial, goles_esperados, probs_1x2, top_marcadores

FECHA_17 = [('Comerciantes Unidos', 'Sport Boys'), ('Juan Pablo II', 'Dep. Garcilaso'), ('UTC', 'Los Chankas'),
            ('CD Moquegua', 'Atlético Grau'), ('Cusco FC', 'ADT'), ('Alianza Atlético', 'Melgar'),
            ('Alianza Lima', 'FC Cajamarca'), ('Sport Huancayo', 'Universitario'), ('Sporting Cristal', 'Cienciano')]


def pronosticar(liga, cruces, sel):
    t = max(p.t for p in liga.partidos) + 1
    test = []
    for h, a in cruces:
        hadv = 1
        if liga.nombre.startswith('Mundial'):
            from datos_mundial2026 import ANFITRIONES
            hadv = 1 if h in ANFITRIONES else (-1 if a in ANFITRIONES else 0)
        test.append(Partido(t, liga.idx(h), liga.idx(a), 0, 0, hadv, 'prox'))
    est = mm.predecir(liga, liga.partidos, test, t, mm.HP(alpha=sel['alpha'], **sel['base']))
    deep = modelo_deep.predecir(liga, liga.partidos, test, t, **VARIANTES_DEEP[sel['deep']])
    w = sel['w']
    return [dict(est=e, deep=d, ens=w * e + (1 - w) * d) for e, d in zip(est, deep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--liga', choices=('liga1', 'mundial'), default='liga1')
    ap.add_argument('--partidos', help='"Local-Visita;Local-Visita" (nombres como en los datos)')
    ap.add_argument('--peso', type=float, help='peso del modelo estadístico en el ensamble (0 a 1); por defecto el del backtest')
    a = ap.parse_args()
    liga = cargar_liga1() if a.liga == 'liga1' else cargar_mundial()
    if not os.path.exists('seleccion.json'):
        raise SystemExit('Falta seleccion.json: ejecuta primero  python backtest.py')
    sel = json.load(open('seleccion.json'))[liga.nombre]
    if a.peso is not None:
        sel['w'] = a.peso
    cruces = [tuple(x.strip() for x in c.split('-', 1)) for c in a.partidos.split(';')] if a.partidos else FECHA_17
    print(f"{liga.nombre} · {len(liga.partidos)} partidos de entrenamiento · estadístico {sel['base']} α={sel['alpha']} · "
          f"deep '{sel['deep']}' · peso estadístico {sel['w']}\n")
    for (h, a_), r in zip(cruces, pronosticar(liga, cruces, sel)):
        M = r['ens']; p = probs_1x2(M); lh, la = goles_esperados(M)
        i, j = np.indices(M.shape)
        tops = '  '.join(f'{x}-{y} ({q:.0%})' for x, y, q in top_marcadores(M, 3))
        pe, pd = probs_1x2(r['est']), probs_1x2(r['deep'])
        print(f'{h} vs {a_}')
        print(f'   1X2: local {p[0]:.0%} · empate {p[1]:.0%} · visita {p[2]:.0%}     goles esperados {lh:.2f} - {la:.2f}')
        print(f'   marcadores más probables: {tops}')
        print(f'   más de 2.5 goles {M[i + j > 2.5].sum():.0%} · ambos anotan {M[(i > 0) & (j > 0)].sum():.0%}')
        print(f'   (solo estadístico {pe[0]:.0%}/{pe[1]:.0%}/{pe[2]:.0%} · solo red neuronal {pd[0]:.0%}/{pd[1]:.0%}/{pd[2]:.0%})\n')


if __name__ == '__main__':
    main()
