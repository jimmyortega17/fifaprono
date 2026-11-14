"""
BACKTEST WALK-FORWARD de todos los modelos sobre los resultados cargados en el repositorio.

Regla única: para predecir la ronda t solo se usan partidos de rondas anteriores. Eso incluye
  - el entrenamiento de cada modelo,
  - la elección de hiperparámetros del modelo mejorado (se elige el que mejor predijo las rondas PASADAS),
  - el peso del ensamble estadístico + deep learning.

Uso:  python backtest.py            (Liga 1 y Mundial; escribe RESULTADOS.md y resultados_backtest.json)
"""
import json
import sys
import time

import numpy as np

import modelo_deep
import modelo_mejorado as mm
from nucleo import MAXG, cargar_liga1, cargar_mundial, dc_matrix, marcador_modal, metricas, probs_1x2, res_idx

VARIANTES_DEEP = {'cero': dict(modo='cero'), 'ajuste': dict(modo='ajuste'), 'ajuste_fuerte': dict(modo='ajuste', ridge=3.0, ridge_h=10.0),
                  'fino': dict(modo='fino')}
MIN_PASADOS = 27      # partidos ya evaluados que se exigen antes de elegir algo con datos


def ll_marc(mats, reales):
    return [-np.log(M[min(gh, MAXG), min(ga, MAXG)]) for M, (gh, ga) in zip(mats, reales)]


def elegir(cands, reales, rondas, defecto):
    """Para cada ronda elige el candidato con menor log-loss de marcador en las rondas anteriores."""
    perd = {k: {t: ll_marc(v[t], reales[t]) for t in rondas} for k, v in cands.items()}
    out, elegido = {}, {}
    for i, t in enumerate(rondas):
        prev = rondas[:i]
        if sum(len(reales[u]) for u in prev) >= MIN_PASADOS:
            k = min(cands, key=lambda c: np.mean([x for u in prev for x in perd[c][u]]))
        else:
            k = defecto
        out[t], elegido[t] = cands[k][t], k
    # elección para la PRÓXIMA ronda (aún no jugada): usa todas las rondas evaluadas
    elegido['siguiente'] = min(cands, key=lambda c: np.mean([x for u in rondas for x in perd[c][u]]))
    return out, elegido


# ------------------------------------------------------------------ modelos de referencia
def base_promedio(train, test):
    if train:
        lh, la = np.mean([p.gh for p in train]), np.mean([p.ga for p in train])
    else:
        lh = la = 1.3
    return [dc_matrix(max(lh, .05), max(la, .05), 0.0) for _ in test]


def original_liga1(liga, train, test, t):
    import modelo_liga1_walkforward as M0
    assert M0.TEAMS == liga.equipos
    f = lambda ps: [(p.t, p.h, p.a, p.gh, p.ga, 'x', 0) for p in ps]
    preds, _ = M0.predict_fecha(f(train), f(test), t)
    return [q['M'] for q in preds], None


def original_mundial(liga, train, test, t):
    """Modelo 2 de auditoria_walkforward.py (Elo dinámico + DC ρ=-0.13 + boost J3 + corrección de goleada)."""
    e = liga.elo0.copy()
    for p in train:
        pw = 1 / (1 + 10 ** ((e[p.a] - e[p.h]) / 400)); s = 1 if p.gh > p.ga else (0.5 if p.gh == p.ga else 0)
        d = 40 * (1 + 0.1 * max(abs(p.gh - p.ga) - 2, 0)) * (s - pw); e[p.h] += d; e[p.a] -= d
    mats, picks = [], []
    for p in test:
        pw = 1 / (1 + 10 ** ((e[p.a] - e[p.h]) / 400)); base = 1.3942 * (1.15 if t == 3 else 1.0)
        l1, l2 = base * (0.5 + pw), base * (1.5 - pw)
        M = dc_matrix(l1, l2, -0.13); b = list(marcador_modal(M))
        if l1 >= 1.75 and l1 - l2 > 0.8: b[0] = max(b[0], round(l1))
        if l2 >= 1.75 and l2 - l1 > 0.8: b[1] = max(b[1], round(l2))
        mats.append(M); picks.append(tuple(b))
    return mats, picks


# ------------------------------------------------------------------------ backtest
def correr(liga, t_ini, original, log=print):
    rondas = [t for t in liga.rondas() if t >= t_ini]
    reales = {t: [(p.gh, p.ga) for p in liga.partidos if p.t == t] for t in rondas}
    est = {}                                   # (i_base, alpha) -> {t: mats}
    fijo, dvar, base, orig, orig_picks = {}, {}, {}, {}, {}
    t0 = time.time()
    for t in rondas:
        train = [p for p in liga.partidos if p.t < t]; test = [p for p in liga.partidos if p.t == t]
        for i, b in enumerate(mm.REJILLA_BASE):
            comp = mm.Componentes(liga, train, t, **b)
            for a in mm.ALPHAS:
                est.setdefault((i, a), {})[t] = comp.matrices(test, a)
        fijo[t] = mm.predecir(liga, train, test, t)
        for nombre, kw in VARIANTES_DEEP.items():
            dvar.setdefault(nombre, {})[t] = modelo_deep.predecir(liga, train, test, t, **kw)
        base[t] = base_promedio(train, test)
        orig[t], pk = original(liga, train, test, t)
        orig_picks[t] = pk
        log(f'  {liga.nombre}: ronda {liga.etiqueta(t)} lista ({time.time() - t0:.0f}s)')
    d = mm.HP_DEFECTO
    i_def = next(i for i, b in enumerate(mm.REJILLA_BASE) if (b['xi'], b['ridge'], b['ridge_h']) == (0.03, 8.0, 30.0))
    auto, hp_elegido = elegir(est, reales, rondas, (i_def, d.alpha))
    deep, deep_elegido = elegir(dvar, reales, rondas, 'ajuste')
    mezclas = {w: {t: [w * a + (1 - w) * b for a, b in zip(auto[t], deep[t])] for t in rondas} for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
    ens, w_elegido = elegir(mezclas, reales, rondas, 0.5)
    modelos = {'Base (promedio de goles)': base, 'Modelo original del repositorio': orig,
               'Estadístico mejorado (HP fijos)': fijo, 'Estadístico mejorado (HP auto)': auto,
               'Deep: red preentrenada, sin efectos de equipo': dvar['cero'], 'Deep learning (afinado, auto)': deep, 'ENSAMBLE estadístico + deep': ens,
               'Ensamble 50/50 fijo': mezclas[0.5]}
    return dict(liga=liga, rondas=rondas, reales=reales, modelos=modelos, orig_picks=orig_picks,
                hp={t: (mm.REJILLA_BASE[k[0]], k[1]) for t, k in hp_elegido.items()}, clave=liga.nombre, w=w_elegido, deep=deep_elegido)


def tabla(res, rondas=None):
    rondas = rondas or res['rondas']
    y = [r for t in rondas for r in res['reales'][t]]
    filas = []
    for nombre, pr in res['modelos'].items():
        mats = [M for t in rondas for M in pr[t]]
        picks = None
        if nombre.startswith('Modelo original') and res['orig_picks'][rondas[0]] is not None:
            picks = [pk for t in rondas for pk in res['orig_picks'][t]]
        filas.append((nombre, metricas(mats, y, picks)))
    return filas


def fmt(filas):
    out = ['| Modelo | Partidos | Acierto 1X2 | Marcador exacto | Marcador en top-3 | LogLoss 1X2 | RPS | LogLoss marcador |',
           '|---|---|---|---|---|---|---|---|']
    for n, m in filas:
        out.append(f"| {n} | {m['n']} | {m['acc']:.1%} | {m['n_exacto']}/{m['n']} ({m['exacto']:.1%}) | {m['top3']:.1%} | "
                   f"{m['logloss']:.3f} | {m['rps']:.3f} | {m['ll_marcador']:.3f} |")
    return '\n'.join(out)


def pareado(res, a, b, n_boot=5000):
    """Diferencia media de log-loss de marcador (a - b) con IC 95 % bootstrap, y duelo de aciertos 1X2."""
    rondas = res['rondas']; y = [r for t in rondas for r in res['reales'][t]]
    A = [M for t in rondas for M in res['modelos'][a][t]]; B = [M for t in rondas for M in res['modelos'][b][t]]
    d = np.array(ll_marc(A, y)) - np.array(ll_marc(B, y))
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (n_boot, len(d))).mean(1)
    yy = np.array([res_idx(*r) for r in y])
    oa = np.array([probs_1x2(M).argmax() for M in A]) == yy; ob = np.array([probs_1x2(M).argmax() for M in B]) == yy
    return dict(dif=float(d.mean()), ic=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))),
                solo_a=int((oa & ~ob).sum()), solo_b=int((~oa & ob).sum()))


def main():
    log = lambda s: print(s, file=sys.stderr, flush=True)
    md = ['# Resultados del backtest walk-forward', '',
          'Generado por `python backtest.py`. Cada ronda se predice usando solo rondas anteriores '
          '(entrenamiento, hiperparámetros y peso del ensamble).', '',
          '- **Acierto 1X2**: acertó local / empate / visita. **Marcador exacto**: el marcador más probable fue el real.',
          '- **LogLoss / RPS**: calidad de las probabilidades; **menor es mejor**. '
          '"LogLoss marcador" mide la probabilidad dada al marcador real.', '']
    salida = {}
    for liga, t_ini, original, sub in ((cargar_liga1(), 6, original_liga1, [28, 29, 30, 31, 32]),
                                       (cargar_mundial(), 1, original_mundial, None)):
        res = correr(liga, t_ini, original, log)
        filas = tabla(res)
        md += [f'## {liga.nombre}', '',
               f'Rondas evaluadas: {liga.etiqueta(res["rondas"][0])} a {liga.etiqueta(res["rondas"][-1])}.', '', fmt(filas), '']
        salida[liga.nombre] = {'global': dict(filas)}
        if sub:
            fs = tabla(res, sub)
            md += ['### Misma ventana que el informe original (Clausura 11 a 15, 45 partidos)', '', fmt(fs), '']
            salida[liga.nombre]['C11-C15'] = dict(fs)
        md += ['### Comparación pareada (mismos partidos)', '']
        ref = 'Modelo original del repositorio'
        for a in ('Estadístico mejorado (HP auto)', 'Deep learning (afinado, auto)', 'ENSAMBLE estadístico + deep'):
            c = pareado(res, a, ref)
            md.append(f"- **{a}** vs original: LogLoss marcador {c['dif']:+.3f} (IC 95 %: {c['ic'][0]:+.3f} a {c['ic'][1]:+.3f}); "
                      f"en 1X2 acierta {c['solo_a']} que el original falla y falla {c['solo_b']} que el original acierta.")
            salida[liga.nombre][f'{a} vs original'] = c
        ult = 'siguiente'
        md += ['', f"Elección para la próxima ronda (la usa `pronosticar.py`): hiperparámetros elegidos {res['hp'][ult][0]}, α={res['hp'][ult][1]}; "
                   f"variante deep elegida = {res['deep'][ult]}; peso del modelo estadístico en el ensamble = {res['w'][ult]}.", '']
        salida[liga.nombre]['hp_ultima'] = dict(base=res['hp'][ult][0], alpha=res['hp'][ult][1], w=res['w'][ult], deep=res['deep'][ult])
    open('RESULTADOS.md', 'w').write('\n'.join(md) + '\n')
    json.dump({k: v['hp_ultima'] for k, v in salida.items()}, open('seleccion.json', 'w'), indent=1, ensure_ascii=False)
    json.dump(salida, open('resultados_backtest.json', 'w'), indent=1, ensure_ascii=False)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
