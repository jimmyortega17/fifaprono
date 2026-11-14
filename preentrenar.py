"""
PREENTRENAMIENTO de la red neuronal con el histórico público de partidos internacionales
(https://github.com/martj42/international_results, licencia CC0).

  - Solo se usan partidos ANTERIORES al 11-jun-2026 (inicio del Mundial): ningún partido del backtest
    del repositorio entra al preentrenamiento.
  - Cada año calendario se trata como una "temporada" y se presenta a la red de dos formas:
      (A) con Elo previo conocido (situación del Mundial) y
      (B) con todos los equipos partiendo de 1500 (situación de Liga 1: sin historia previa).
  - Entrenamiento: 1994-2023. Validación (para parar a tiempo): 2024 a 10-jun-2026.
  - Se guardan varias redes (semillas distintas) en pesos/preentrenado.pt.

Uso:  python preentrenar.py        (descarga el CSV con git si no está en datos_externos/)
"""
import csv
import os
import subprocess
import sys
from datetime import date

import numpy as np
import torch

import modelo_deep as D
from nucleo import Liga, Partido, elo_delta

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(AQUI, 'datos_externos', 'international_results', 'results.csv')
CORTE = date(2026, 6, 11)
ANIO_INI, ANIO_VAL = 1994, 2024
K_ELO, HFA = 30.0, 60.0
SEMILLAS, OCULTO, DROPOUT, EPOCAS, LOTE = 5, 64, 0.10, 25, 1024


def leer():
    if not os.path.exists(CSV):
        os.makedirs(os.path.dirname(os.path.dirname(CSV)), exist_ok=True)
        subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/martj42/international_results',
                        os.path.dirname(CSV)], check=True)
    filas = []
    for r in csv.DictReader(open(CSV, encoding='utf-8')):
        try:
            f = date.fromisoformat(r['date'])
            gh, ga = int(r['home_score']), int(r['away_score'])
        except ValueError:
            continue
        if f < CORTE and f.year >= 1950:
            filas.append((f, r['home_team'], r['away_team'], gh, ga, 0 if r['neutral'].upper() == 'TRUE' else 1))
    filas.sort(key=lambda x: x[0])
    return filas


def temporadas(filas):
    """Genera (año, Liga, partidos) en versión A (Elo previo) y B (todos en 1500)."""
    equipos = sorted({f[1] for f in filas} | {f[2] for f in filas})
    ix = {e: i for i, e in enumerate(equipos)}
    ref = Liga('ref', equipos, [], np.full(len(equipos), 1500.0), K_ELO, HFA)
    R = ref.elo0.copy()
    por_anio = {}
    for f in filas:
        por_anio.setdefault(f[0].year, []).append(f)
    for anio in sorted(por_anio):
        ps = [Partido(f[0].toordinal(), ix[f[1]], ix[f[2]], f[3], f[4], f[5], str(anio)) for f in por_anio[anio]]
        if anio >= ANIO_INI:
            yield anio, Liga(f'{anio}A', equipos, ps, R.copy(), K_ELO, HFA), ps
            yield anio, Liga(f'{anio}B', equipos, ps, np.full(len(equipos), 1500.0), K_ELO, HFA), ps
        for p in ps:
            d = elo_delta(ref, R[p.h], R[p.a], p.hadv, p.gh, p.ga)
            R[p.h] += d
            R[p.a] -= d


def main():
    filas = leer()
    print(f'{len(filas)} partidos internacionales desde 1950 hasta {filas[-1][0]} (corte {CORTE})')
    tr, va = [], []
    for anio, liga, ps in temporadas(filas):
        f, _ = D.construir(liga, ps)
        (va if anio >= ANIO_VAL else tr).extend(f)
    Xtr, _, _, hadvtr, ghtr, gatr, _ = D.tensores(tr)
    Xva, _, _, hadvva, ghva, gava, _ = D.tensores(va)
    print(f'filas de entrenamiento {len(tr)}, validación {len(va)}')
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-3)

    def nll(red, X, hadv, gh, ga):
        eh, ea = red(X, hadv)
        return -D.log_ver(eh, ea, red.rho, gh, ga).mean()

    # referencia: el prior Elo sin red
    red0 = D.Nucleo(OCULTO, DROPOUT); red0.eval()
    with torch.no_grad():
        print(f'validación, prior sin entrenar: {nll(red0, Xva, hadvva, ghva, gava).item():.4f}')
    estados = []
    for s in range(SEMILLAS):
        torch.manual_seed(s)
        red = D.Nucleo(OCULTO, DROPOUT)
        red.mu.copy_(mu); red.sd.copy_(sd)
        opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-3)
        plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)
        mejor, mejor_sd = 1e9, None
        for ep in range(EPOCAS):
            red.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), LOTE):
                b = perm[i:i + LOTE]
                opt.zero_grad()
                nll(red, Xtr[b], hadvtr[b], ghtr[b], gatr[b]).backward()
                opt.step()
            plan.step()
            red.eval()
            with torch.no_grad():
                v = nll(red, Xva, hadvva, ghva, gava).item()
            if v < mejor:
                mejor, mejor_sd = v, {k: x.clone() for k, x in red.state_dict().items()}
        print(f'semilla {s}: mejor log-verosimilitud negativa de validación = {mejor:.4f}', flush=True)
        estados.append(mejor_sd)
    os.makedirs(os.path.dirname(D.RUTA_PESOS), exist_ok=True)
    torch.save(dict(estados=estados, oculto=OCULTO, dropout=DROPOUT, corte=str(CORTE)), D.RUTA_PESOS)
    print('guardado', D.RUTA_PESOS)


if __name__ == '__main__':
    sys.exit(main())
