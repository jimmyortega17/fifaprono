"""
PRONÓSTICO MUNDIAL 2026 — partidos faltantes de fase de grupos
Modelo: fuerzas de ataque/defensa ajustadas por rival (Poisson iterativo)
        + goles ~ Poisson (función aleatoria) + Monte Carlo + Random Forest.
Autor: generado para Jimmy (INVERSIONES ORBUS)
"""
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

RNG = np.random.default_rng(2026)
K_SHRINK = 3.0          # "partidos virtuales" de regularización (jala fuerzas hacia 1.0)
N_ITER   = 60           # iteraciones del ajuste de fuerzas
N_SIMS   = 1000         # simulaciones Monte Carlo por partido

# ----------------------------------------------------------------------
# 1) DATOS: partidos jugados (J1+J2 todos; J3 de A,B,C,E) — oficiales FIFA al 25-jun-2026
#    formato: (local, goles_local, goles_visita, visita, grupo)
# ----------------------------------------------------------------------
PLAYED = [
 ("MEX",2,0,"RSA","A"),("KOR",2,1,"CZE","A"),("CZE",1,1,"RSA","A"),("MEX",1,0,"KOR","A"),("CZE",0,3,"MEX","A"),("RSA",1,0,"KOR","A"),
 ("CAN",1,1,"BIH","B"),("QAT",1,1,"SUI","B"),("SUI",4,1,"BIH","B"),("CAN",6,0,"QAT","B"),("SUI",2,1,"CAN","B"),("BIH",3,1,"QAT","B"),
 ("BRA",1,1,"MAR","C"),("HAI",0,1,"SCO","C"),("SCO",0,1,"MAR","C"),("BRA",3,0,"HAI","C"),("SCO",0,3,"BRA","C"),("MAR",4,2,"HAI","C"),
 ("USA",4,1,"PAR","D"),("AUS",2,0,"TUR","D"),("USA",2,0,"AUS","D"),("TUR",0,1,"PAR","D"),
 ("GER",7,1,"CUW","E"),("CIV",1,0,"ECU","E"),("GER",2,1,"CIV","E"),("ECU",0,0,"CUW","E"),("CUW",0,2,"CIV","E"),("ECU",2,1,"GER","E"),
 ("NED",2,2,"JPN","F"),("SWE",5,1,"TUN","F"),("NED",5,1,"SWE","F"),("TUN",0,4,"JPN","F"),
 ("BEL",1,1,"EGY","G"),("IRN",2,2,"NZL","G"),("BEL",0,0,"IRN","G"),("NZL",1,3,"EGY","G"),
 ("ESP",0,0,"CPV","H"),("KSA",1,1,"URU","H"),("ESP",4,0,"KSA","H"),("URU",2,2,"CPV","H"),
 ("FRA",3,1,"SEN","I"),("IRQ",1,4,"NOR","I"),("FRA",3,0,"IRQ","I"),("NOR",3,2,"SEN","I"),
 ("ARG",3,0,"ALG","J"),("AUT",3,1,"JOR","J"),("ARG",2,0,"AUT","J"),("JOR",1,2,"ALG","J"),
 ("POR",1,1,"COD","K"),("UZB",1,3,"COL","K"),("POR",5,0,"UZB","K"),("COL",1,0,"COD","K"),
 ("ENG",4,2,"CRO","L"),("GHA",1,0,"PAN","L"),("ENG",0,0,"GHA","L"),("CRO",1,0,"PAN","L"),
]

# Partidos faltantes (J3 de los grupos D,F,G,H,I,J,K,L)
REMAINING = [
 ("D","TUR","USA"),("D","PAR","AUS"),
 ("F","JPN","SWE"),("F","TUN","NED"),
 ("G","EGY","IRN"),("G","NZL","BEL"),
 ("H","CPV","KSA"),("H","URU","ESP"),
 ("I","NOR","FRA"),("I","SEN","IRQ"),
 ("J","ALG","AUT"),("J","JOR","ARG"),
 ("K","COL","POR"),("K","COD","UZB"),
 ("L","PAN","ENG"),("L","CRO","GHA"),
]

NAME = {"MEX":"México","RSA":"Sudáfrica","KOR":"Corea","CZE":"Chequia","CAN":"Canadá","BIH":"Bosnia",
"QAT":"Catar","SUI":"Suiza","BRA":"Brasil","MAR":"Marruecos","HAI":"Haití","SCO":"Escocia","USA":"EE.UU.",
"PAR":"Paraguay","AUS":"Australia","TUR":"Turquía","GER":"Alemania","CUW":"Curazao","CIV":"C. de Marfil",
"ECU":"Ecuador","NED":"P. Bajos","JPN":"Japón","SWE":"Suecia","TUN":"Túnez","BEL":"Bélgica","EGY":"Egipto",
"IRN":"Irán","NZL":"N. Zelanda","ESP":"España","CPV":"Cabo Verde","KSA":"Arabia S.","URU":"Uruguay",
"FRA":"Francia","SEN":"Senegal","IRQ":"Irak","NOR":"Noruega","ARG":"Argentina","ALG":"Argelia","AUT":"Austria",
"JOR":"Jordania","POR":"Portugal","COD":"R.D. Congo","COL":"Colombia","UZB":"Uzbekistán","ENG":"Inglaterra",
"CRO":"Croacia","GHA":"Ghana","PAN":"Panamá"}

# ----------------------------------------------------------------------
# 2) ESTIMACIÓN DE FUERZAS (ataque/defensa) ajustadas por rival
# ----------------------------------------------------------------------
def fit_strengths(matches, k=K_SHRINK, iters=N_ITER):
    teams = sorted({m[0] for m in matches} | {m[3] for m in matches})
    total_goals = sum(m[1]+m[2] for m in matches)
    L = total_goals / (2*len(matches))            # goles promedio por equipo/partido
    att = {t:1.0 for t in teams}; dfn = {t:1.0 for t in teams}
    for _ in range(iters):
        na, nd = {}, {}
        for t in teams:
            gf_num=gf_den=ga_num=ga_den=0.0
            for h,hg,ag,a,_ in matches:
                if t==h:
                    gf_num+=hg; gf_den+=L*dfn[a]; ga_num+=ag; ga_den+=L*att[a]
                elif t==a:
                    gf_num+=ag; gf_den+=L*dfn[h]; ga_num+=hg; ga_den+=L*att[h]
            na[t]=(gf_num+k)/(gf_den+k)            # shrink hacia 1.0
            nd[t]=(ga_num+k)/(ga_den+k)
        # normalizar a media 1
        ma=np.mean(list(na.values())); md=np.mean(list(nd.values()))
        att={t:na[t]/ma for t in teams}; dfn={t:nd[t]/md for t in teams}
    return att, dfn, L

# ----------------------------------------------------------------------
# 3) FUNCIÓN DE PRONÓSTICO  (goles ~ Poisson = la función aleatoria)
# ----------------------------------------------------------------------
def expected_goals(local, visita, att, dfn, L):
    lam_l = L * att[local] * dfn[visita]
    lam_v = L * att[visita] * dfn[local]
    return lam_l, lam_v

def outcome_probs(lam_l, lam_v, max_g=10):
    """Probabilidades analíticas 1/X/2 con doble Poisson."""
    pl = poisson.pmf(np.arange(max_g+1), lam_l)
    pv = poisson.pmf(np.arange(max_g+1), lam_v)
    M  = np.outer(pl, pv)
    p1 = np.tril(M,-1).sum(); px = np.trace(M); p2 = np.triu(M,1).sum()
    modal = np.unravel_index(np.argmax(M), M.shape)   # marcador más probable
    return p1, px, p2, (int(modal[0]), int(modal[1]))

def pronosticar(local, visita, att, dfn, L, n_sims=N_SIMS, rng=RNG):
    """
    FUNCIÓN DE PRONÓSTICO.
    Devuelve probabilidades 1/X/2, goles esperados, marcador modal,
    el equipo más débil y un marcador simulado (Monte Carlo).
    """
    lam_l, lam_v = expected_goals(local, visita, att, dfn, L)
    p1, px, p2, modal = outcome_probs(lam_l, lam_v)
    # Monte Carlo (la "función aleatoria"): goles ~ Poisson
    gl = rng.poisson(lam_l, n_sims); gv = rng.poisson(lam_v, n_sims)
    mc_modal = Counter(zip(gl.tolist(), gv.tolist())).most_common(1)[0][0]
    poder_l = att[local]/dfn[local]; poder_v = att[visita]/dfn[visita]
    debil = visita if poder_l>=poder_v else local
    pick = local if p1==max(p1,px,p2) else (visita if p2==max(p1,px,p2) else "Empate")
    return {
        "local":local,"visita":visita,"lam_l":lam_l,"lam_v":lam_v,
        "p1":p1,"px":px,"p2":p2,"modal":modal,"mc_modal":mc_modal,
        "mas_debil":debil,"pronostico":pick,
        "sim_gl":gl,"sim_gv":gv,
    }

# ----------------------------------------------------------------------
# 4) BACKTESTING del modelo Poisson (leave-one-out, re-estimando fuerzas)
# ----------------------------------------------------------------------
def label(hg,ag): return "H" if hg>ag else ("A" if ag>hg else "D")

def backtest_poisson(matches):
    correct=0; logloss=0.0; n=len(matches)
    for i in range(n):
        train=[m for j,m in enumerate(matches) if j!=i]
        a,d,L = fit_strengths(train)
        h,hg,ag,aw,_ = matches[i]
        if h not in a or aw not in a:   # equipo sin datos en train -> saltar
            n-=1; continue
        ll,lv = expected_goals(h,aw,a,d,L)
        p1,px,p2,_ = outcome_probs(ll,lv)
        probs={"H":p1,"D":px,"A":p2}
        pred=max(probs,key=probs.get); real=label(hg,ag)
        correct += (pred==real)
        logloss += -np.log(max(probs[real],1e-9))
    return correct/n, logloss/n, n

# ----------------------------------------------------------------------
# 5) RANDOM FOREST  (eficacia clasificando 1/X/2)
# ----------------------------------------------------------------------
def build_features(matches, att, dfn, L):
    X=[]; y=[]
    for h,hg,ag,a,_ in matches:
        ll,lv = expected_goals(h,a,att,dfn,L)
        X.append([att[h],dfn[h],att[a],dfn[a],
                  att[h]-att[a], dfn[a]-dfn[h],
                  ll, lv, ll-lv, (att[h]/dfn[h])-(att[a]/dfn[a])])
        y.append(label(hg,ag))
    return np.array(X), np.array(y)

FEATURES = ["atk_L","def_L","atk_V","def_V","dif_atk","dif_def","xg_L","xg_V","dif_xg","dif_poder"]

# ======================================================================
# EJECUCIÓN
# ======================================================================
att, dfn, L = fit_strengths(PLAYED)
print(f"L (goles prom/equipo) = {L:.3f}")

# Backtest Poisson
acc_p, ll_p, n_bt = backtest_poisson(PLAYED)
# Baseline: gana el de mayor poder (sin empate)
base_ok=0
for h,hg,ag,a,_ in PLAYED:
    pred = "H" if (att[h]/dfn[h])>=(att[a]/dfn[a]) else "A"
    base_ok += (pred==label(hg,ag))
acc_base = base_ok/len(PLAYED)

# Random Forest
X,y = build_features(PLAYED, att, dfn, L)
rf = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=2, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
rf.fit(X,y)
acc_rf_in = rf.score(X,y)
cm = confusion_matrix(y, rf.predict(X), labels=["H","D","A"])
imp = dict(zip(FEATURES, rf.feature_importances_))

print(f"Backtest Poisson LOO  acc={acc_p:.3f}  logloss={ll_p:.3f}  (n={n_bt})")
print(f"Baseline 'gana el mejor' acc={acc_base:.3f}")
print(f"Random Forest  CV acc={cv.mean():.3f} ± {cv.std():.3f}  | in-sample={acc_rf_in:.3f}")

# Pronóstico de los 16 partidos faltantes
pred_rows=[]
sim_matrix=[]   # 1000 x 16 marcadores
for grupo,h,a in REMAINING:
    r = pronosticar(h,a,att,dfn,L)
    Xr = np.array([[att[h],dfn[h],att[a],dfn[a],att[h]-att[a],dfn[a]-dfn[h],
                    r["lam_l"],r["lam_v"],r["lam_l"]-r["lam_v"],(att[h]/dfn[h])-(att[a]/dfn[a])]])
    rf_proba = rf.predict_proba(Xr)[0]
    rf_cls = rf.classes_
    rf_pick = rf_cls[np.argmax(rf_proba)]
    rf_pick_txt = {"H":NAME[h],"D":"Empate","A":NAME[a]}[rf_pick]
    pred_rows.append({
        "grupo":grupo,"local":h,"visita":a,"debil":r["mas_debil"],
        "p1":r["p1"],"px":r["px"],"p2":r["p2"],
        "xg_l":r["lam_l"],"xg_v":r["lam_v"],
        "modal":f'{r["modal"][0]}-{r["modal"][1]}',
        "pronostico":r["pronostico"],"rf_pick":rf_pick_txt,
        "rf_prob":float(np.max(rf_proba)),
    })
    sim_matrix.append([f"{gl}-{gv}" for gl,gv in zip(r["sim_gl"],r["sim_gv"])])

sim_matrix=np.array(sim_matrix).T   # 1000 filas x 16 cols

import pickle
with open("model_out.pkl","wb") as f:
    pickle.dump(dict(att=att,dfn=dfn,L=L,acc_p=acc_p,ll_p=ll_p,acc_base=acc_base,
                     cv=cv,acc_rf_in=acc_rf_in,cm=cm,imp=imp,pred_rows=pred_rows,
                     sim_matrix=sim_matrix,REMAINING=REMAINING,PLAYED=PLAYED,NAME=NAME,FEATURES=FEATURES),f)
print("Modelo OK. Partidos faltantes:",len(REMAINING))
