"""
Modelo Poisson-Dixon-Coles + Elo para Liga 1 Perú 2026, validación walk-forward.
Para cada fecha de prueba k (Clausura 11..15) se entrena SOLO con partidos anteriores
(Apertura completo disponible + Clausura 1..k-1). Hiperparámetros fijados a priori (no ajustados en el test).
"""
import sys, json
import numpy as np
from math import factorial
from scipy.optimize import minimize, minimize_scalar
sys.path.insert(0, '/home/claude/liga1')
from datos_liga1_2026 import PARTIDOS

# ---------------- hiperparámetros fijos ----------------
ALPHA = 0.50      # peso del modelo att/def en el ensamble convexo
RIDGE = 8.0       # penalización (contracción) de att/def hacia 0
XI = 0.02         # decaimiento temporal por fecha (peso = exp(-XI * antigüedad_en_fechas))
K_ELO = 20.0      # K de Elo
HFA_ELO = 60.0    # ventaja de local en puntos Elo
MAXG = 10         # goles máximos en la matriz

TEAMS = sorted({p[2] for p in PARTIDOS} | {p[3] for p in PARTIDOS})
IDX = {t: i for i, t in enumerate(TEAMS)}
N = len(TEAMS)

# índice temporal secuencial: Apertura 1..17 -> 1..17 ; Clausura f -> 17+f
def tidx(torneo, f):
    return f if torneo == 'A' else 17 + f

DATA = [(tidx(t, f), IDX[h], IDX[a], gh, ga, t, f) for (t, f, h, a, gh, ga) in PARTIDOS]
DATA.sort(key=lambda r: r[0])


def pois_pmf(l, n=MAXG + 1):
    k = np.arange(n)
    return np.exp(-l + k * np.log(max(l, 1e-9)) - np.array([np.log(factorial(int(i))) for i in k]))


def tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def dc_matrix(lh, la, rho):
    M = np.outer(pois_pmf(lh), pois_pmf(la))
    for x in (0, 1):
        for y in (0, 1):
            M[x, y] *= tau(x, y, lh, la, rho)
    M = np.clip(M, 0, None)
    return M / M.sum()


# ---------------- Elo dinámico ----------------
def run_elo(train):
    R = np.full(N, 1500.0)
    for r in sorted(train, key=lambda r: r[0]):
        _, h, a, gh, ga = r[:5]
        E = 1 / (1 + 10 ** (-(R[h] + HFA_ELO - R[a]) / 400))
        S = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        gd = abs(gh - ga)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        d = K_ELO * mult * (S - E)
        R[h] += d; R[a] -= d
    return R


def fit_elo_map(train, R_hist):
    """Ajusta (g_h, g_a, c) del mapa Elo -> lambdas con ratings previos a cada partido (sin look-ahead)."""
    R = np.full(N, 1500.0)
    rows = []
    for r in sorted(train, key=lambda r: r[0]):
        _, h, a, gh, ga = r[:5]
        dE = (R[h] + HFA_ELO - R[a]) / 400
        rows.append((dE, gh, ga))
        E = 1 / (1 + 10 ** (-(R[h] + HFA_ELO - R[a]) / 400))
        S = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        gd = abs(gh - ga)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        d = K_ELO * mult * (S - E)
        R[h] += d; R[a] -= d
    rows = np.array(rows)
    dE, gh, ga = rows[:, 0], rows[:, 1], rows[:, 2]

    def nll(p):
        lgh, lga, c = p
        lh = np.exp(lgh + c * dE); la = np.exp(lga - c * dE)
        return -(gh * np.log(lh) - lh + ga * np.log(la) - la).sum()
    res = minimize(nll, [np.log(1.4), np.log(1.0), 0.3], method='Nelder-Mead')
    return res.x


def elo_lambdas(R, h, a, par):
    lgh, lga, c = par
    dE = (R[h] + HFA_ELO - R[a]) / 400
    return np.exp(lgh + c * dE), np.exp(lga - c * dE)


# ---------------- att/def penalizado con decaimiento ----------------
def fit_attdef(train, k_now):
    H = np.array([r[1] for r in train]); A = np.array([r[2] for r in train])
    GH = np.array([r[3] for r in train], float); GA = np.array([r[4] for r in train], float)
    w = np.exp(-XI * (k_now - np.array([r[0] for r in train], float)))

    def unpack(p):
        return p[0], p[1], p[2:2 + N], p[2 + N:2 + 2 * N]

    def f(p):
        mu, hfa, att, de = unpack(p)
        eh = mu + hfa + att[H] - de[A]
        ea = mu + att[A] - de[H]
        lh, la = np.exp(eh), np.exp(ea)
        ll = (w * (GH * eh - lh + GA * ea - la)).sum()
        pen = RIDGE * ((att ** 2).sum() + (de ** 2).sum())
        # gradiente
        rh = w * (GH - lh); ra = w * (GA - la)
        g = np.zeros_like(p)
        g[0] = (rh + ra).sum(); g[1] = rh.sum()
        np.add.at(g, 2 + H, rh); np.add.at(g, 2 + A, ra)
        np.add.at(g, 2 + N + A, -rh); np.add.at(g, 2 + N + H, -ra)
        g[2:2 + N] -= 2 * RIDGE * att; g[2 + N:] -= 2 * RIDGE * de
        return -(ll - pen), -g
    p0 = np.zeros(2 + 2 * N); p0[0] = np.log(1.1)
    res = minimize(f, p0, jac=True, method='L-BFGS-B')
    return unpack(res.x)


def attdef_lambdas(par, h, a):
    mu, hfa, att, de = par
    return np.exp(mu + hfa + att[h] - de[a]), np.exp(mu + att[a] - de[h])


def fit_rho(train, lam_fn):
    """rho de Dixon-Coles por MLE sobre el set de entrenamiento (lambdas del ensamble)."""
    obs = []
    for r in train:
        lh, la = lam_fn(r[1], r[2])
        obs.append((r[3], r[4], lh, la))

    def nll(rho):
        s = 0.0
        for x, y, lh, la in obs:
            t = tau(x, y, lh, la, rho)
            if t <= 0: return 1e9
            s -= np.log(t)  # los factores Poisson no dependen de rho
        return s
    res = minimize_scalar(nll, bounds=(-0.3, 0.2), method='bounded')
    return res.x


def probs_1x2(M):
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])  # H, D, A


def predict_fecha(train, test, k_now):
    Rf = run_elo(train)
    pmap = fit_elo_map(train, None)
    ad = fit_attdef(train, k_now)

    def lam(h, a):
        le = elo_lambdas(Rf, h, a, pmap); lm = attdef_lambdas(ad, h, a)
        return (1 - ALPHA) * le[0] + ALPHA * lm[0], (1 - ALPHA) * le[1] + ALPHA * lm[1]
    rho = fit_rho(train, lam)
    out = []
    for r in test:
        h, a = r[1], r[2]
        lh, la = lam(h, a)
        le = elo_lambdas(Rf, h, a, pmap)
        M = dc_matrix(lh, la, rho)
        Me = dc_matrix(le[0], le[1], rho)
        out.append(dict(h=TEAMS[h], a=TEAMS[a], gh=r[3], ga=r[4], lh=lh, la=la, rho=rho,
                        p=probs_1x2(M), p_elo=probs_1x2(Me), M=M))
    return out, rho


def result_idx(gh, ga):
    return 0 if gh > ga else (1 if gh == ga else 2)


def metrics(preds, key='p'):
    n = len(preds)
    P = np.array([q[key] for q in preds]); y = np.array([result_idx(q['gh'], q['ga']) for q in preds])
    acc = (P.argmax(1) == y).mean()
    ll = -np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).mean()
    Y = np.eye(3)[y]
    brier = ((P - Y) ** 2).sum(1).mean()
    cumP = np.cumsum(P, 1)[:, :2]; cumY = np.cumsum(Y, 1)[:, :2]
    rps = ((cumP - cumY) ** 2).sum(1).mean() / 2
    return dict(n=n, acc=acc, logloss=ll, brier=brier, rps=rps)


if __name__ == '__main__':
    TEST_FECHAS = [11, 12, 13, 14, 15]   # Clausura
    allp = []; perfecha = {}
    for f in TEST_FECHAS:
        k = tidx('C', f)
        train = [r for r in DATA if r[0] < k]
        test = [r for r in DATA if r[0] == k]
        preds, rho = predict_fecha(train, test, k)
        for q in preds: q['fecha'] = f
        perfecha[f] = (preds, rho, len(train))
        allp += preds
    # baseline climatológico: frecuencias H/D/A del entrenamiento previo a cada fecha
    base = []
    for f in TEST_FECHAS:
        k = tidx('C', f)
        tr = [r for r in DATA if r[0] < k]
        fr = np.bincount([result_idx(r[3], r[4]) for r in tr], minlength=3) / len(tr)
        for q in perfecha[f][0]:
            base.append(dict(gh=q['gh'], ga=q['ga'], p=fr))
    # ---- reporte ----
    print('FECHA  n  acc   logloss  brier   rps   |  MAE_goles_tot  sesgo_goles  rho   n_train')
    for f in TEST_FECHAS:
        ps, rho, nt = perfecha[f]
        m = metrics(ps)
        tot_err = np.mean([abs((q['lh'] + q['la']) - (q['gh'] + q['ga'])) for q in ps])
        bias = np.mean([(q['lh'] + q['la']) - (q['gh'] + q['ga']) for q in ps])
        print(f'C{f:<4} {m["n"]:>2} {m["acc"]:.3f}  {m["logloss"]:.3f}   {m["brier"]:.3f}  {m["rps"]:.3f}  |  {tot_err:.2f}          {bias:+.2f}       {rho:+.3f} {nt}')
    mm = metrics(allp); me = metrics(allp, 'p_elo'); mb = metrics(base)
    print('\nGLOBAL (45 partidos)')
    for nm, m in (('Ensamble PDC (modelo)', mm), ('Solo Elo+DC', me), ('Base frecuencias', mb)):
        print(f'  {nm:<24} acc={m["acc"]:.3f} logloss={m["logloss"]:.3f} brier={m["brier"]:.3f} rps={m["rps"]:.3f}')
    gh_err = np.array([q['lh'] - q['gh'] for q in allp]); ga_err = np.array([q['la'] - q['ga'] for q in allp])
    print(f'  MAE goles local={np.abs(gh_err).mean():.3f} visita={np.abs(ga_err).mean():.3f}; RMSE local={np.sqrt((gh_err**2).mean()):.3f} visita={np.sqrt((ga_err**2).mean()):.3f}')
    print(f'  Goles totales: pronosticado medio={np.mean([q["lh"]+q["la"] for q in allp]):.2f} real medio={np.mean([q["gh"]+q["ga"] for q in allp]):.2f}')
    ex = sum(1 for q in allp if np.unravel_index(q['M'].argmax(), q['M'].shape) == (q['gh'], q['ga']))
    print(f'  Marcador exacto acertado (modo): {ex}/{len(allp)}')
    # detalle
    rows = []
    for q in allp:
        i, j = np.unravel_index(q['M'].argmax(), q['M'].shape)
        pr = ['L', 'E', 'V'][int(np.argmax(q['p']))]; re_ = ['L', 'E', 'V'][result_idx(q['gh'], q['ga'])]
        rows.append(dict(fecha=q['fecha'], partido=f"{q['h']} - {q['a']}", real=f"{q['gh']}-{q['ga']}",
                         pron=f"{i}-{j}", lam=f"{q['lh']:.2f}-{q['la']:.2f}",
                         pL=round(float(q['p'][0]), 3), pE=round(float(q['p'][1]), 3), pV=round(float(q['p'][2]), 3),
                         pick=pr, res=re_, ok=int(pr == re_)))
    json.dump(dict(rows=rows, global_=dict(modelo=mm, elo=me, base=mb)), open('/home/claude/liga1/resultados.json', 'w'),
              default=float, ensure_ascii=False, indent=1)
    for r in rows:
        print(f"C{r['fecha']} {r['partido']:<42} real {r['real']}  pron {r['pron']}  λ {r['lam']}  L/E/V {r['pL']:.2f}/{r['pE']:.2f}/{r['pV']:.2f}  {r['pick']}->{r['res']} {'OK' if r['ok'] else 'x'}")
