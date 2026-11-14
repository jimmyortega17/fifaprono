"""
Compara varios modelos de pronóstico 1X2 sobre los mismos 45 partidos (Clausura C11..C15), walk-forward:
cada fecha se predice solo con información previa. Hiperparámetros fijos, sin ajuste con el test.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, '/home/claude/liga1')
warnings.filterwarnings('ignore')
import modelo_liga1_walkforward as M
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA = M.DATA; N = M.N; TEAMS = M.TEAMS
res = M.result_idx

# ---------- features pre-partido (solo pasado) ----------
def build_features():
    pts = np.zeros(N); gf = np.zeros(N); ga = np.zeros(N); g = np.zeros(N)
    last = [[] for _ in range(N)]
    R = np.full(N, 1500.0)
    rows = {}
    for k in sorted({r[0] for r in DATA}):
        batch = [r for r in DATA if r[0] == k]
        for r in batch:
            _, h, a, gh, ga_, *_ = r
            def f(t):
                n = max(g[t], 1)
                l5 = last[t][-5:]
                return [pts[t] / n, (gf[t] - ga[t]) / n, gf[t] / n, ga[t] / n,
                        (np.mean(l5) if l5 else 1.0)]
            fh, fa = f(h), f(a)
            x = [R[h] - R[a]] + [u - v for u, v in zip(fh, fa)]
            rows[id(r)] = (x, min(g[h], g[a]))
        for r in batch:
            _, h, a, gh, ga_, *_ = r
            E = 1 / (1 + 10 ** (-(R[h] + M.HFA_ELO - R[a]) / 400))
            S = 1.0 if gh > ga_ else 0.5 if gh == ga_ else 0.0
            gd = abs(gh - ga_); mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
            d = M.K_ELO * mult * (S - E); R[h] += d; R[a] -= d
            ph = 3 if gh > ga_ else 1 if gh == ga_ else 0
            pa = 3 if ga_ > gh else 1 if gh == ga_ else 0
            pts[h] += ph; pts[a] += pa; gf[h] += gh; ga[h] += ga_; gf[a] += ga_; ga[a] += gh; g[h] += 1; g[a] += 1
            last[h].append(ph / 3); last[a].append(pa / 3)
    return rows

FEAT = build_features()

def XY(rs):
    X = np.array([FEAT[id(r)][0] for r in rs]); y = np.array([res(r[3], r[4]) for r in rs])
    return X, y

def metr(P, y):
    n = len(y); P = np.clip(np.asarray(P), 1e-6, 1); P = P / P.sum(1, keepdims=True)
    acc = (P.argmax(1) == y).mean()
    ll = -np.log(P[np.arange(n), y]).mean()
    Y = np.eye(3)[y]
    brier = ((P - Y) ** 2).sum(1).mean()
    rps = ((np.cumsum(P, 1)[:, :2] - np.cumsum(Y, 1)[:, :2]) ** 2).sum(1).mean() / 2
    return acc, ll, brier, rps

TEST = [11, 12, 13, 14, 15]
store = {}   # nombre -> lista de probs ; y
ys = []
for f in TEST:
    k = M.tidx('C', f)
    train = [r for r in DATA if r[0] < k]; test = [r for r in DATA if r[0] == k]
    ys += [res(r[3], r[4]) for r in test]
    fr = np.bincount([res(r[3], r[4]) for r in train], minlength=3) / len(train)
    # --- modelos del proyecto
    Rf = M.run_elo(train); pmap = M.fit_elo_map(train, None); ad = M.fit_attdef(train, k)
    out = {}
    def lam_mix(a_):
        def lam(h, a):
            le = M.elo_lambdas(Rf, h, a, pmap); lm = M.attdef_lambdas(ad, h, a)
            return (1 - a_) * le[0] + a_ * lm[0], (1 - a_) * le[1] + a_ * lm[1]
        return lam
    for nm, a_ in (('Solo Elo (Poisson+DC)', 0.0), ('Solo ataque/defensa (Poisson+DC)', 1.0), ('MI MODELO (ensamble 50/50)', 0.5)):
        lam = lam_mix(a_); rho = M.fit_rho(train, lam)
        out[nm] = [M.probs_1x2(M.dc_matrix(*lam(r[1], r[2]), rho)) for r in test]
    # --- baselines triviales
    out['Siempre local'] = [np.array([1, 0, 0]) * 0.98 + 0.01 for r in test]
    out['Frecuencias históricas'] = [fr for r in test]
    # --- ML sobre features (Elo + puntos + dif. de gol + forma)
    trn = [r for r in train if FEAT[id(r)][1] >= 3]
    Xtr, ytr = XY(trn); Xte, _ = XY(test)
    mods = {
        'Regresión logística (Elo+forma)': make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
        'Gradient Boosting (Elo+forma)': HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.05, random_state=0),
        'Random Forest (Elo+forma)': RandomForestClassifier(300, min_samples_leaf=8, random_state=0),
    }
    for nm, mdl in mods.items():
        mdl.fit(Xtr, ytr); out[nm] = list(mdl.predict_proba(Xte))
    # --- regla simple de tabla: gana el de más puntos por partido (+0.35 local), empate si casi igual
    sc = []
    for r in test:
        x = FEAT[id(r)][0]; d = x[1] + 0.35
        p = np.array([1 / (1 + np.exp(-2.2 * d)) * 0.8 + 0.03, 0.22, 1 / (1 + np.exp(2.2 * d)) * 0.8 + 0.03])
        sc.append(p / p.sum())
    out['Regla de tabla (puntos/partido)'] = sc
    for nm, v in out.items():
        store.setdefault(nm, []).extend(v)

y = np.array(ys)
rows = []
for nm, P in store.items():
    a, ll, b, rp = metr(P, y)
    hits = int(round(a * len(y)))
    rows.append((nm, hits, a, ll, b, rp))
rows.sort(key=lambda r: -r[1] * 10 + r[3])
print(f"{'Modelo':<38}{'Aciertos':>9}{'Acc':>7}{'LogLoss':>9}{'Brier':>8}{'RPS':>8}")
for nm, h, a, ll, b, rp in rows:
    print(f"{nm:<38}{h:>6}/45{a:>7.1%}{ll:>9.3f}{b:>8.3f}{rp:>8.3f}")

# comparación pareada con mi modelo
mine = np.array(store['MI MODELO (ensamble 50/50)']).argmax(1) == y
print('\nComparación pareada vs MI MODELO (partidos donde uno acierta y el otro no):')
for nm, P in store.items():
    if nm.startswith('MI '): continue
    o = np.array(P).argmax(1) == y
    print(f"  {nm:<38} yo gano {int((mine & ~o).sum()):>2} | ellos ganan {int((~mine & o).sum()):>2}")
