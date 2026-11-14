"""
MUNDIAL 2026 — v2
(1) Ajuste de función aleatoria estilo Arena Input Analyzer (min. error cuadrático, Chi2, K-S)
(2) Selección del modelo de pronóstico por mejor AUC (OvR macro)
(3) Pronóstico de partidos faltantes + 1000 simulaciones con la mejor distribución
"""
import numpy as np, pickle
from collections import Counter
from scipy import stats
from scipy.stats import poisson, nbinom, geom
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score

RNG=np.random.default_rng(2026); K_SHRINK=3.0; N_ITER=60; N_SIMS=1000

PLAYED=[("MEX",2,0,"RSA","A"),("KOR",2,1,"CZE","A"),("CZE",1,1,"RSA","A"),("MEX",1,0,"KOR","A"),("CZE",0,3,"MEX","A"),("RSA",1,0,"KOR","A"),
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
("ENG",4,2,"CRO","L"),("GHA",1,0,"PAN","L"),("ENG",0,0,"GHA","L"),("CRO",1,0,"PAN","L")]
REMAINING=[("D","TUR","USA"),("D","PAR","AUS"),("F","JPN","SWE"),("F","TUN","NED"),("G","EGY","IRN"),("G","NZL","BEL"),
("H","CPV","KSA"),("H","URU","ESP"),("I","NOR","FRA"),("I","SEN","IRQ"),("J","ALG","AUT"),("J","JOR","ARG"),
("K","COL","POR"),("K","COD","UZB"),("L","PAN","ENG"),("L","CRO","GHA")]
NAME={"MEX":"México","RSA":"Sudáfrica","KOR":"Corea","CZE":"Chequia","CAN":"Canadá","BIH":"Bosnia","QAT":"Catar","SUI":"Suiza","BRA":"Brasil","MAR":"Marruecos","HAI":"Haití","SCO":"Escocia","USA":"EE.UU.","PAR":"Paraguay","AUS":"Australia","TUR":"Turquía","GER":"Alemania","CUW":"Curazao","CIV":"C. de Marfil","ECU":"Ecuador","NED":"P. Bajos","JPN":"Japón","SWE":"Suecia","TUN":"Túnez","BEL":"Bélgica","EGY":"Egipto","IRN":"Irán","NZL":"N. Zelanda","ESP":"España","CPV":"Cabo Verde","KSA":"Arabia S.","URU":"Uruguay","FRA":"Francia","SEN":"Senegal","IRQ":"Irak","NOR":"Noruega","ARG":"Argentina","ALG":"Argelia","AUT":"Austria","JOR":"Jordania","POR":"Portugal","COD":"R.D. Congo","COL":"Colombia","UZB":"Uzbekistán","ENG":"Inglaterra","CRO":"Croacia","GHA":"Ghana","PAN":"Panamá"}

# ====================================================================
# (1) INPUT ANALYZER — ajuste de distribución a los goles observados
# ====================================================================
goals = np.array([g for h,hg,ag,a,_ in PLAYED for g in (hg,ag)])  # 112 observaciones
N=len(goals); mu=goals.mean(); var=goals.var(ddof=1)
KMAX=7  # bins 0..6 y cola >=7
edges=list(range(KMAX))   # 0..6
def hist_rel(data):
    c=np.zeros(KMAX)
    for v in data:
        c[min(v,KMAX-1)]+=1
    return c/len(data)
obs_rel=hist_rel(goals)
obs_cnt=obs_rel*N

def pmf_binned(pmf_func):
    p=np.array([pmf_func(k) for k in range(KMAX-1)])
    p=np.append(p, max(1-p.sum(),0))   # último bin = cola
    return p

# candidatas (estilo Input Analyzer; discretas para datos de conteo)
cands={}
cands["Poisson"]=dict(par=f"λ={mu:.3f}", pmf=lambda k,l=mu: poisson.pmf(k,l), k=1)
# Negative Binomial por momentos (sobredispersión)
if var>mu:
    r=mu**2/(var-mu); p_nb=r/(r+mu)
    cands["Neg. Binomial"]=dict(par=f"r={r:.2f}, p={p_nb:.3f}", pmf=lambda k,r=r,p=p_nb: nbinom.pmf(k,r,p), k=2)
else:
    r=np.inf
# Geométrica en {0,1,...}
pg=1/(1+mu)
cands["Geométrica"]=dict(par=f"p={pg:.3f}", pmf=lambda k,p=pg: p*(1-p)**k, k=1)
# Uniforme discreta [0, max]
mx=int(goals.max())
cands["Unif. discreta"]=dict(par=f"[0,{mx}]", pmf=lambda k,m=mx: 1/(m+1) if k<=m else 0, k=1)

fit_table=[]
for nm,info in cands.items():
    p=pmf_binned(info["pmf"])
    sq=float(np.sum((obs_rel-p)**2))               # ERROR CUADRÁTICO (criterio Arena)
    E=p*N
    mask=E>=5                                        # Chi2 agrupando E<5
    chi=float(np.sum((obs_cnt[mask]-E[mask])**2/E[mask])) if mask.sum()>1 else np.nan
    df=max(mask.sum()-1-info["k"],1)
    pchi=float(1-stats.chi2.cdf(chi,df)) if not np.isnan(chi) else np.nan
    # K-S discreto
    Femp=np.cumsum(obs_rel); Ffit=np.cumsum(p); ks=float(np.max(np.abs(Femp-Ffit)))
    fit_table.append(dict(dist=nm,par=info["par"],sqerr=sq,chi=chi,pchi=pchi,ks=ks))
fit_table.sort(key=lambda d:d["sqerr"])
best_dist=fit_table[0]["dist"]

# Sobredispersión condicional (Var/Media de residuos respecto a la media del modelo)
# (se evalúa luego con las fuerzas; aquí guardamos ratio global)
disp_global=var/mu

# ====================================================================
# Estimación de fuerzas (media multiplicativa, agnóstica a la familia)
# ====================================================================
def fit_strengths(matches,k=K_SHRINK,iters=N_ITER):
    teams=sorted({m[0] for m in matches}|{m[3] for m in matches})
    L=sum(m[1]+m[2] for m in matches)/(2*len(matches))
    att={t:1.0 for t in teams}; dfn={t:1.0 for t in teams}
    for _ in range(iters):
        na,nd={},{}
        for t in teams:
            gfn=gfd=gan=gad=0.0
            for h,hg,ag,a,_ in matches:
                if t==h: gfn+=hg; gfd+=L*dfn[a]; gan+=ag; gad+=L*att[a]
                elif t==a: gfn+=ag; gfd+=L*dfn[h]; gan+=hg; gad+=L*att[h]
            na[t]=(gfn+k)/(gfd+k); nd[t]=(gan+k)/(gad+k)
        ma=np.mean(list(na.values())); md=np.mean(list(nd.values()))
        att={t:na[t]/ma for t in teams}; dfn={t:nd[t]/md for t in teams}
    return att,dfn,L
att,dfn,L=fit_strengths(PLAYED)
def xg(h,a,att,dfn,L): return L*att[h]*dfn[a], L*att[a]*dfn[h]

# dispersión condicional: residuos (g - lambda) estandarizados
resid_disp=[]
for h,hg,ag,a,_ in PLAYED:
    lh,la=xg(h,a,att,dfn,L); resid_disp+=[(hg-lh),(ag-la)]
# r condicional global por momentos sobre lambdas
lams=[]; gobs=[]
for h,hg,ag,a,_ in PLAYED:
    lh,la=xg(h,a,att,dfn,L); lams+=[lh,la]; gobs+=[hg,ag]
lams=np.array(lams); gobs=np.array(gobs)
# var condicional aproximada
cond_var=np.mean((gobs-lams)**2); cond_mean=np.mean(lams)
r_cond = (cond_mean**2/(cond_var-cond_mean)) if cond_var>cond_mean else np.inf

# Familia para la SIMULACIÓN: si hay sobredispersión condicional -> NegBin, si no -> Poisson
USE_NB = np.isfinite(r_cond) and (cond_var > 1.15*cond_mean)
SIM_FAMILY = "Neg. Binomial" if USE_NB else "Poisson"

def prob_vector(lh,la,maxg=12):
    if USE_NB:
        ph=nbinom.pmf(np.arange(maxg+1), r_cond, r_cond/(r_cond+lh))
        pa=nbinom.pmf(np.arange(maxg+1), r_cond, r_cond/(r_cond+la))
    else:
        ph=poisson.pmf(np.arange(maxg+1),lh); pa=poisson.pmf(np.arange(maxg+1),la)
    M=np.outer(ph,pa)
    p1=np.tril(M,-1).sum(); px=np.trace(M); p2=np.triu(M,1).sum()
    modal=np.unravel_index(np.argmax(M),M.shape)
    s=p1+px+p2
    return p1/s,px/s,p2/s,(int(modal[0]),int(modal[1]))

def sim_goals(lam,n,rng=RNG):
    if USE_NB: return rng.negative_binomial(r_cond, r_cond/(r_cond+lam), n)
    return rng.poisson(lam,n)

def label(hg,ag): return "H" if hg>ag else ("A" if ag>hg else "D")

# ====================================================================
# (2) SELECCIÓN POR AUC — bake-off de modelos (OvR macro)
# ====================================================================
def features(matches,att,dfn,L):
    X=[];y=[]
    for h,hg,ag,a,_ in matches:
        lh,la=xg(h,a,att,dfn,L)
        X.append([att[h],dfn[h],att[a],dfn[a],att[h]-att[a],dfn[a]-dfn[h],lh,la,lh-la,(att[h]/dfn[h])-(att[a]/dfn[a])])
        y.append(label(hg,ag))
    return np.array(X),np.array(y)
FEATURES=["atk_L","def_L","atk_V","def_V","dif_atk","dif_def","xg_L","xg_V","dif_xg","dif_poder"]
X,y=features(PLAYED,att,dfn,L)
classes=["H","D","A"]; Yb=label_binarize(y,classes=classes)

def auc_macro(Yb,P): return roc_auc_score(Yb,P,average="macro",multi_class="ovr")

# Modelo analítico (Poisson/NB) por LOO (out-of-sample)
P_analytic=np.zeros((len(PLAYED),3)); pred_an=[]
for i in range(len(PLAYED)):
    tr=[m for j,m in enumerate(PLAYED) if j!=i]
    a2,d2,L2=fit_strengths(tr)
    h,hg,ag,aw,_=PLAYED[i]
    lh,la=xg(h,aw,a2,d2,L2); p1,px,p2,_=prob_vector(lh,la)
    P_analytic[i]=[p1,px,p2]; pred_an.append(["H","D","A"][int(np.argmax([p1,px,p2]))])
auc_an=auc_macro(Yb,P_analytic); acc_an=accuracy_score(y,pred_an)

# Modelos ML por CV (out-of-sample, predict_proba)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
models={
 "Reg. Logística":LogisticRegression(max_iter=2000,C=1.0),
 "Random Forest":RandomForestClassifier(n_estimators=400,max_depth=4,min_samples_leaf=2,random_state=42),
 "Gradient Boosting":GradientBoostingClassifier(random_state=42),
 "Naive Bayes":GaussianNB(),
}
results=[{"modelo":f"Analítico ({SIM_FAMILY})","auc":auc_an,"acc":acc_an}]
proba_store={f"Analítico ({SIM_FAMILY})":P_analytic}
for nm,mdl in models.items():
    proba=cross_val_predict(mdl,X,y,cv=skf,method="predict_proba")
    order=list(mdl.fit(X,y).classes_)
    Pord=proba[:,[order.index(c) for c in classes]]
    proba_store[nm]=Pord
    a_=auc_macro(Yb,Pord); ac_=accuracy_score(y,[classes[i] for i in np.argmax(Pord,axis=1)])
    results.append({"modelo":nm,"auc":a_,"acc":ac_})
results.sort(key=lambda d:-d["auc"])
best_model_name=results[0]["modelo"]

# Reentrenar el mejor modelo sobre todo para pronosticar (si es ML), CALIBRADO
from sklearn.calibration import CalibratedClassifierCV
best_is_ml = best_model_name in models
CALIBRATED=False; auc_cal=None
if best_is_ml:
    base=models[best_model_name]
    cal_cv=CalibratedClassifierCV(base, method="sigmoid", cv=skf)
    proba_cal=cross_val_predict(cal_cv,X,y,cv=skf,method="predict_proba")
    ord_c=list(CalibratedClassifierCV(base,method="sigmoid",cv=skf).fit(X,y).classes_)
    Pcal=proba_cal[:,[ord_c.index(c) for c in classes]]
    auc_cal=auc_macro(Yb,Pcal)
    best_est=CalibratedClassifierCV(base, method="sigmoid", cv=5).fit(X,y)
    best_order=list(best_est.classes_); CALIBRATED=True

# ====================================================================
# (3) PRONÓSTICO + 1000 SIMULACIONES con la mejor distribución
# ====================================================================
pred_rows=[]; sim_matrix=[]
for grupo,h,a in REMAINING:
    lh,la=xg(h,a,att,dfn,L); p1,px,p2,modal=prob_vector(lh,la)
    if best_is_ml:
        Xr=np.array([[att[h],dfn[h],att[a],dfn[a],att[h]-att[a],dfn[a]-dfn[h],lh,la,lh-la,(att[h]/dfn[h])-(att[a]/dfn[a])]])
        pr=best_est.predict_proba(Xr)[0]; pr={c:pr[best_order.index(c)] for c in classes}
        p1m,pxm,p2m=pr["H"],pr["D"],pr["A"]
    else:
        p1m,pxm,p2m=p1,px,p2
    pick=NAME[h] if max(p1m,pxm,p2m)==p1m else (NAME[a] if max(p1m,pxm,p2m)==p2m else "Empate")
    debil=a if att[h]/dfn[h]>=att[a]/dfn[a] else h
    gl=sim_goals(lh,N_SIMS); gv=sim_goals(la,N_SIMS)
    sim_matrix.append([f"{x}-{y_}" for x,y_ in zip(gl.tolist(),gv.tolist())])
    pred_rows.append(dict(grupo=grupo,local=h,visita=a,debil=debil,p1=p1m,px=pxm,p2=p2m,
                          xg_l=lh,xg_v=la,modal=f"{modal[0]}-{modal[1]}",pick=pick))
sim_matrix=np.array(sim_matrix).T

pickle.dump(dict(fit_table=fit_table,best_dist=best_dist,goals=goals,obs_rel=obs_rel,cands=list(cands.keys()),
    mu=mu,var=var,disp_global=disp_global,cond_var=cond_var,cond_mean=cond_mean,r_cond=r_cond,
    SIM_FAMILY=SIM_FAMILY,results=results,best_model_name=best_model_name,FEATURES=FEATURES,CALIBRATED=CALIBRATED,auc_cal=auc_cal,
    pred_rows=pred_rows,sim_matrix=sim_matrix,REMAINING=REMAINING,NAME=NAME,att=att,dfn=dfn,L=L,
    pmf_best={k:[float(cands[k]["pmf"](i)) for i in range(8)] for k in cands}),
    open("model2.pkl","wb"))

print("=== INPUT ANALYZER (orden por error cuadrático) ===")
for f in fit_table: print(f"{f['dist']:<16}{f['par']:<22} sqErr={f['sqerr']:.5f}  Chi2={f['chi']:.2f} p={f['pchi']:.3f}  KS={f['ks']:.3f}")
print(f"\nMejor ajuste (datos crudos): {best_dist}")
print(f"Dispersión global Var/Media={disp_global:.2f} | condicional Var/Media={cond_var/cond_mean:.2f} -> r={r_cond:.2f}")
print(f"Familia usada en simulación: {SIM_FAMILY}")
print("\n=== BAKE-OFF POR AUC (OvR macro, out-of-sample) ===")
for r in results: print(f"{r['modelo']:<24} AUC={r['auc']:.3f}  acc={r['acc']*100:.1f}%")
print(f"\nMejor modelo (AUC): {best_model_name}")
