"""
Mercados derivados de la matriz Poisson-DC: 1X2, doble oportunidad, goles totales (Over/Under),
ambos anotan, goles por equipo, marcador exacto y combinadas. Incluye:
  (1) calibración en backtest C11..C15 (¿las probabilidades altas realmente aciertan?)
  (2) pronóstico de los partidos pendientes de la fecha 16.
No hay datos de jugadores, así que NO hay mercado de goleadores.
"""
import sys
import numpy as np
sys.path.insert(0, '/home/claude/liga1')
import modelo_liga1_walkforward as M

MAXG = M.MAXG


def markets(Mx):
    n = Mx.shape[0]
    i, j = np.indices(Mx.shape)
    tot = i + j
    P = {}
    P['Local gana'] = Mx[i > j].sum(); P['Empate'] = Mx[i == j].sum(); P['Visita gana'] = Mx[i < j].sum()
    P['1X (local o empate)'] = P['Local gana'] + P['Empate']
    P['X2 (visita o empate)'] = P['Visita gana'] + P['Empate']
    P['12 (sin empate)'] = P['Local gana'] + P['Visita gana']
    for L in (0.5, 1.5, 2.5, 3.5, 4.5):
        P[f'Más de {L}'] = Mx[tot > L].sum(); P[f'Menos de {L}'] = Mx[tot < L].sum()
    P['Ambos anotan SI'] = Mx[(i >= 1) & (j >= 1)].sum(); P['Ambos anotan NO'] = 1 - P['Ambos anotan SI']
    P['Local anota'] = Mx[i >= 1].sum(); P['Visita anota'] = Mx[j >= 1].sum()
    # combinadas (probabilidad conjunta real desde la matriz, no producto)
    P['1X + Más de 0.5'] = Mx[(i >= j) & (tot > 0.5)].sum()
    P['1X + Más de 1.5'] = Mx[(i >= j) & (tot > 1.5)].sum()
    P['X2 + Más de 0.5'] = Mx[(i <= j) & (tot > 0.5)].sum()
    P['X2 + Más de 1.5'] = Mx[(i <= j) & (tot > 1.5)].sum()
    P['Local gana + Más de 1.5'] = Mx[(i > j) & (tot > 1.5)].sum()
    P['Visita gana + Más de 1.5'] = Mx[(i < j) & (tot > 1.5)].sum()
    P['1X + Menos de 3.5'] = Mx[(i >= j) & (tot < 3.5)].sum()
    P['X2 + Menos de 3.5'] = Mx[(i <= j) & (tot < 3.5)].sum()
    P['Más de 0.5 + Menos de 4.5'] = Mx[(tot > 0.5) & (tot < 4.5)].sum()
    P['Más de 1.5 + Menos de 4.5'] = Mx[(tot > 1.5) & (tot < 4.5)].sum()
    return P


def outcome(name, gh, ga):
    t = gh + ga
    d = {
        'Local gana': gh > ga, 'Empate': gh == ga, 'Visita gana': gh < ga,
        '1X (local o empate)': gh >= ga, 'X2 (visita o empate)': gh <= ga, '12 (sin empate)': gh != ga,
        'Ambos anotan SI': gh >= 1 and ga >= 1, 'Ambos anotan NO': not (gh >= 1 and ga >= 1),
        'Local anota': gh >= 1, 'Visita anota': ga >= 1,
        '1X + Más de 0.5': gh >= ga and t > 0.5, '1X + Más de 1.5': gh >= ga and t > 1.5,
        'X2 + Más de 0.5': gh <= ga and t > 0.5, 'X2 + Más de 1.5': gh <= ga and t > 1.5,
        'Local gana + Más de 1.5': gh > ga and t > 1.5, 'Visita gana + Más de 1.5': gh < ga and t > 1.5,
        '1X + Menos de 3.5': gh >= ga and t < 3.5, 'X2 + Menos de 3.5': gh <= ga and t < 3.5,
        'Más de 0.5 + Menos de 4.5': 0.5 < t < 4.5, 'Más de 1.5 + Menos de 4.5': 1.5 < t < 4.5,
    }
    if name in d: return bool(d[name])
    if name.startswith('Más de '): return t > float(name.split()[-1])
    if name.startswith('Menos de '): return t < float(name.split()[-1])
    raise KeyError(name)


def backtest():
    out = []
    for f in [11, 12, 13, 14, 15]:
        k = M.tidx('C', f)
        train = [r for r in M.DATA if r[0] < k]; test = [r for r in M.DATA if r[0] == k]
        preds, _ = M.predict_fecha(train, test, k)
        out += preds
    return out


if __name__ == '__main__':
    bt = backtest()
    names = list(markets(bt[0]['M']).keys())
    print('=== (1) CALIBRACIÓN EN 45 PARTIDOS (C11-C15) ===')
    print(f"{'Mercado':<30}{'P media modelo':>15}{'Acierto real':>14}{'Mín P':>8}")
    for nm in names:
        ps = np.array([markets(q['M'])[nm] for q in bt]); hit = np.array([outcome(nm, q['gh'], q['ga']) for q in bt])
        print(f"{nm:<30}{ps.mean():>14.1%}{hit.mean():>14.1%}{ps.min():>8.2f}")
    print('\nPartidos individuales con P>=85% en cualquier mercado (backtest) y si acertaron:')
    tot = ok = 0; byband = {}
    for q in bt:
        P = markets(q['M'])
        for nm, p in P.items():
            if p >= 0.85:
                h = outcome(nm, q['gh'], q['ga']); tot += 1; ok += h
    print(f'  selecciones con P>=85%: {tot} -> acertadas {ok} ({ok / max(tot, 1):.1%})')
    for lo, hi in ((0.80, 0.85), (0.85, 0.90), (0.90, 1.01)):
        t = o = 0
        for q in bt:
            for nm, p in markets(q['M']).items():
                if lo <= p < hi:
                    t += 1; o += outcome(nm, q['gh'], q['ga'])
        print(f'  banda {lo:.0%}-{min(hi, 1):.0%}: n={t:>3} aciertos={o:>3} ({o / max(t, 1):.1%})')

    # ===== (2) fecha 16 pendiente =====
    print('\n=== (2) FECHA 16 — pendientes ===')
    k = M.tidx('C', 16) + 1  # decaimiento medido desde después de la f16
    train = [r for r in M.DATA]  # incluye Grau 3-0 Cusco (C16)
    PEND = [('ADT', 'Sporting Cristal'), ('FC Cajamarca', 'Alianza Atlético'), ('Melgar', 'Juan Pablo II'),
            ('Cienciano', 'Sport Huancayo'), ('Los Chankas', 'Alianza Lima'), ('Sport Boys', 'UTC'),
            ('Dep. Garcilaso', 'Comerciantes Unidos'), ('Universitario', 'CD Moquegua')]
    test = [(k, M.IDX[h], M.IDX[a], 0, 0, 'C', 16) for h, a in PEND]
    preds, rho = M.predict_fecha(train, test, k)
    import json
    res = []
    for q in preds:
        P = markets(q['M']); Mx = q['M']
        i, j = np.unravel_index(Mx.argmax(), Mx.shape)
        top = sorted(((Mx[a, b], f'{a}-{b}') for a in range(6) for b in range(6)), reverse=True)[:3]
        print(f"\n{q['h']} vs {q['a']}  (goles esperados {q['lh']:.2f}-{q['la']:.2f})")
        print(f"  1X2: L {P['Local gana']:.0%} | E {P['Empate']:.0%} | V {P['Visita gana']:.0%}   marcador más probable {i}-{j} ({Mx[i, j]:.0%}); top3: " + ', '.join(f'{s} {p:.0%}' for p, s in top))
        best = sorted(P.items(), key=lambda kv: -kv[1])[:6]
        print('  Mercados más seguros: ' + ' | '.join(f'{n} {p:.0%}' for n, p in best))
        res.append(dict(partido=f"{q['h']} vs {q['a']}", lam=[q['lh'], q['la']], P={k_: float(v) for k_, v in P.items()},
                        marcador=f'{i}-{j}', p_marcador=float(Mx[i, j])))
    json.dump(res, open('/home/claude/liga1/pronostico_f16.json', 'w'), ensure_ascii=False, indent=1)
