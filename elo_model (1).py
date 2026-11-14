"""
CORRECCIÓN DEL MODELO — ancla con Elo pre-torneo (eliminatorias + historial)
- Prior de ataque/defensa derivado del Elo pre-Mundial (worldcupelo.com).
- Las fuerzas se estiman anclando hacia el prior; los 3 partidos del Mundial solo ajustan.
- Backtest comparativo: SOLO FORMA vs FORMA+PRIOR (LOO accuracy y AUC).
- Re-simula el torneo y reescribe Prob. Título + nueva hoja Equivalencias.
"""
import numpy as np, pickle
from copy import copy
from scipy.stats import poisson
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

RNG=np.random.default_rng(7); N_SIMS=6000; N_ITER=80
KP=4.0            # peso del prior (≈ partidos virtuales)
BETA=0.0020       # escala Elo -> fuerza de gol

# Elo pre-torneo (worldcupelo.com, jun-2026)
ELO={"ESP":2171,"ARG":2113,"FRA":2063,"ENG":2042,"COL":1998,"BRA":1979,"POR":1976,"NED":1959,
"CRO":1933,"ECU":1933,"NOR":1922,"GER":1910,"SUI":1897,"URU":1890,"TUR":1880,"JPN":1879,"SEN":1869,
"BEL":1849,"MEX":1834,"PAR":1833,"AUT":1818,"MAR":1806,"CAN":1806,"SCO":1790,"KOR":1784,"AUS":1774,
"IRN":1754,"USA":1747,"PAN":1743,"UZB":1735,"CZE":1731,"ALG":1728,"JOR":1691,"SWE":1660,"EGY":1660,
"COD":1639,"CIV":1637,"TUN":1614,"KSA":1592,"NZL":1586,"IRQ":1583,"BIH":1571,"CPV":1561,"HAI":1542,
"RSA":1529,"GHA":1509,"CUW":1467,"QAT":1427}

PLAYED=[("MEX",2,0,"RSA"),("KOR",2,1,"CZE"),("CZE",1,1,"RSA"),("MEX",1,0,"KOR"),("CZE",0,3,"MEX"),("RSA",1,0,"KOR"),
("CAN",1,1,"BIH"),("QAT",1,1,"SUI"),("SUI",4,1,"BIH"),("CAN",6,0,"QAT"),("SUI",2,1,"CAN"),("BIH",3,1,"QAT"),
("BRA",1,1,"MAR"),("HAI",0,1,"SCO"),("SCO",0,1,"MAR"),("BRA",3,0,"HAI"),("SCO",0,3,"BRA"),("MAR",4,2,"HAI"),
("USA",4,1,"PAR"),("AUS",2,0,"TUR"),("USA",2,0,"AUS"),("TUR",0,1,"PAR"),
("GER",7,1,"CUW"),("CIV",1,0,"ECU"),("GER",2,1,"CIV"),("ECU",0,0,"CUW"),("CUW",0,2,"CIV"),("ECU",2,1,"GER"),
("NED",2,2,"JPN"),("SWE",5,1,"TUN"),("NED",5,1,"SWE"),("TUN",0,4,"JPN"),
("BEL",1,1,"EGY"),("IRN",2,2,"NZL"),("BEL",0,0,"IRN"),("NZL",1,3,"EGY"),
("ESP",0,0,"CPV"),("KSA",1,1,"URU"),("ESP",4,0,"KSA"),("URU",2,2,"CPV"),
("FRA",3,1,"SEN"),("IRQ",1,4,"NOR"),("FRA",3,0,"IRQ"),("NOR",3,2,"SEN"),
("ARG",3,0,"ALG"),("AUT",3,1,"JOR"),("ARG",2,0,"AUT"),("JOR",1,2,"ALG"),
("POR",1,1,"COD"),("UZB",1,3,"COL"),("POR",5,0,"UZB"),("COL",1,0,"COD"),
("ENG",4,2,"CRO"),("GHA",1,0,"PAN"),("ENG",0,0,"GHA"),("CRO",1,0,"PAN")]
REMAINING=[("D","TUR","USA"),("D","PAR","AUS"),("F","JPN","SWE"),("F","TUN","NED"),("G","EGY","IRN"),("G","NZL","BEL"),
("H","CPV","KSA"),("H","URU","ESP"),("I","NOR","FRA"),("I","SEN","IRQ"),("J","ALG","AUT"),("J","JOR","ARG"),
("K","COL","POR"),("K","COD","UZB"),("L","PAN","ENG"),("L","CRO","GHA")]
GROUPS={"A":["MEX","RSA","KOR","CZE"],"B":["CAN","SUI","BIH","QAT"],"C":["BRA","MAR","SCO","HAI"],
"D":["USA","PAR","AUS","TUR"],"E":["GER","CIV","ECU","CUW"],"F":["NED","JPN","SWE","TUN"],
"G":["BEL","EGY","IRN","NZL"],"H":["ESP","CPV","KSA","URU"],"I":["FRA","SEN","IRQ","NOR"],
"J":["ARG","AUT","ALG","JOR"],"K":["POR","COL","COD","UZB"],"L":["ENG","GHA","CRO","PAN"]}
GOF={t:g for g,ts in GROUPS.items() for t in ts}
NAME={"MEX":"México","RSA":"Sudáfrica","KOR":"Corea","CZE":"Chequia","CAN":"Canadá","BIH":"Bosnia","QAT":"Catar","SUI":"Suiza","BRA":"Brasil","MAR":"Marruecos","HAI":"Haití","SCO":"Escocia","USA":"EE.UU.","PAR":"Paraguay","AUS":"Australia","TUR":"Turquía","GER":"Alemania","CUW":"Curazao","CIV":"C. de Marfil","ECU":"Ecuador","NED":"P. Bajos","JPN":"Japón","SWE":"Suecia","TUN":"Túnez","BEL":"Bélgica","EGY":"Egipto","IRN":"Irán","NZL":"N. Zelanda","ESP":"España","CPV":"Cabo Verde","KSA":"Arabia S.","URU":"Uruguay","FRA":"Francia","SEN":"Senegal","IRQ":"Irak","NOR":"Noruega","ARG":"Argentina","ALG":"Argelia","AUT":"Austria","JOR":"Jordania","POR":"Portugal","COD":"R.D. Congo","COL":"Colombia","UZB":"Uzbekistán","ENG":"Inglaterra","CRO":"Croacia","GHA":"Ghana","PAN":"Panamá"}

# Prior ataque/defensa desde Elo
elo_bar=np.mean([ELO[t] for t in GOF])
power0={t:np.exp(BETA*(ELO[t]-elo_bar)) for t in GOF}
A0={t:power0[t]**0.5 for t in GOF}; D0={t:power0[t]**-0.5 for t in GOF}
mA=np.mean(list(A0.values())); mD=np.mean(list(D0.values()))
A0={t:A0[t]/mA for t in GOF}; D0={t:D0[t]/mD for t in GOF}

def fit_strengths(matches, kp=KP, iters=N_ITER, use_prior=True):
    teams=sorted({m[0] for m in matches}|{m[3] for m in matches})
    L=sum(m[1]+m[2] for m in matches)/(2*len(matches))
    att={t:1.0 for t in teams}; dfn={t:1.0 for t in teams}
    a0=A0 if use_prior else {t:1.0 for t in teams}
    d0=D0 if use_prior else {t:1.0 for t in teams}
    for _ in range(iters):
        na,nd={},{}
        for t in teams:
            gfn=gfd=gan=gad=0.0
            for h,hg,ag,a in matches:
                if t==h: gfn+=hg; gfd+=L*dfn[a]; gan+=ag; gad+=L*att[a]
                elif t==a: gfn+=ag; gfd+=L*dfn[h]; gan+=hg; gad+=L*att[h]
            na[t]=(gfn+kp*L*a0[t])/(gfd+kp*L)        # ancla hacia el prior Elo
            nd[t]=(gan+kp*L*d0[t])/(gad+kp*L)
        ma=np.mean(list(na.values())); md=np.mean(list(nd.values()))
        att={t:na[t]/ma for t in teams}; dfn={t:nd[t]/md for t in teams}
    return att,dfn,L

ATT,DFN,L=fit_strengths(PLAYED)
def lams(a,b): return L*ATT[a]*DFN[b], L*ATT[b]*DFN[a]

# ---------- BACKTEST comparativo (LOO) ----------
def label(hg,ag): return "H" if hg>ag else ("A" if ag>hg else "D")
def outcome(lh,la,maxg=10):
    ph=poisson.pmf(np.arange(maxg+1),lh); pa=poisson.pmf(np.arange(maxg+1),la)
    M=np.outer(ph,pa); p1=np.tril(M,-1).sum(); px=np.trace(M); p2=np.triu(M,1).sum()
    s=p1+px+p2; return p1/s,px/s,p2/s
def backtest(use_prior):
    y=[]; P=[]
    for i in range(len(PLAYED)):
        tr=[m for j,m in enumerate(PLAYED) if j!=i]
        a,d,L2=fit_strengths(tr,use_prior=use_prior)
        h,hg,ag,aw=PLAYED[i]
        lh,la=L2*a[h]*d[aw], L2*a[aw]*d[h]
        P.append(list(outcome(lh,la))); y.append(label(hg,ag))
    P=np.array(P); y=np.array(y); cls=["H","D","A"]
    acc=np.mean([cls[i] for i in P.argmax(1)]==y)
    auc=roc_auc_score(label_binarize(y,classes=cls),P[:,[0,1,2]],average="macro",multi_class="ovr")
    return acc,auc
acc_form,auc_form=backtest(False)
acc_prior,auc_prior=backtest(True)
print(f"SOLO FORMA   : acc={acc_form*100:.1f}%  AUC={auc_form:.3f}")
print(f"FORMA+PRIOR  : acc={acc_prior*100:.1f}%  AUC={auc_prior:.3f}")

# ---------- Simulación de torneo ----------
base_pts={t:0 for t in GOF}; base_gf={t:0 for t in GOF}; base_ga={t:0 for t in GOF}
for h,hg,ag,a in PLAYED:
    base_gf[h]+=hg; base_ga[h]+=ag; base_gf[a]+=ag; base_ga[a]+=hg
    if hg>ag: base_pts[h]+=3
    elif ag>hg: base_pts[a]+=3
    else: base_pts[h]+=1; base_pts[a]+=1
R32=[("2A","2B"),("1C","2F"),("1E","T"),("1F","2C"),("2E","2I"),("1I","T"),("1A","T"),("1L","T"),
     ("1D","2J"),("1H","2L"),("2K","T"),("1B","2G"),("2D","2H"),("1J","T"),("1K","T"),("1G","T")]
THIRD_SLOTS=[2,5,6,7,10,13,14,15]; THIRD_EXCL={2:"E",5:"I",6:"A",7:"L",10:"K",13:"J",14:"K",15:"G"}
def pm(a,b): la,lb=lams(a,b); return RNG.poisson(la),RNG.poisson(lb)
def ko(a,b):
    ga,gb=pm(a,b)
    if ga>gb: return a
    if gb>ga: return b
    pa=ATT[a]/DFN[a]; pb=ATT[b]/DFN[b]; return a if RNG.random()<pa/(pa+pb) else b
Rcnt={t:dict(qual=0,r16=0,qf=0,sf=0,fin=0,champ=0) for t in GOF}
for s in range(N_SIMS):
    pts=dict(base_pts); gf=dict(base_gf); ga=dict(base_ga)
    for g,h,a in REMAINING:
        hg,ag=pm(h,a); gf[h]+=hg; ga[h]+=ag; gf[a]+=ag; ga[a]+=hg
        if hg>ag: pts[h]+=3
        elif ag>hg: pts[a]+=3
        else: pts[h]+=1; pts[a]+=1
    pos={}; thirds=[]
    for g,ts in GROUPS.items():
        rk=sorted(ts,key=lambda t:(pts[t],gf[t]-ga[t],gf[t],RNG.random()),reverse=True)
        pos[f"1{g}"]=rk[0]; pos[f"2{g}"]=rk[1]; t3=rk[2]; thirds.append((t3,pts[t3],gf[t3]-ga[t3],gf[t3]))
    thirds.sort(key=lambda x:(x[1],x[2],x[3],RNG.random()),reverse=True); q3=[t for t,_,_,_ in thirds[:8]]
    for t in [pos[f"1{g}"] for g in GROUPS]+[pos[f"2{g}"] for g in GROUPS]+q3: Rcnt[t]["qual"]+=1
    cost=np.zeros((8,8))
    for i,t in enumerate(q3):
        for j,slot in enumerate(THIRD_SLOTS): cost[i,j]=1e6 if GOF[t]==THIRD_EXCL[slot] else RNG.random()
    ri,cj=linear_sum_assignment(cost); slot_team={THIRD_SLOTS[j]:q3[i] for i,j in zip(ri,cj)}
    res=lambda slot,idx: slot_team[idx] if slot=="T" else pos[slot]
    teams=[(res(a,i),res(b,i)) for i,(a,b) in enumerate(R32)]
    w=[ko(a,b) for a,b in teams]
    for t in w: Rcnt[t]["r16"]+=1
    qf=[ko(w[i],w[i+1]) for i in range(0,16,2)]
    for t in qf: Rcnt[t]["qf"]+=1
    sf=[ko(qf[i],qf[i+1]) for i in range(0,8,2)]
    for t in sf: Rcnt[t]["sf"]+=1
    fn=[ko(sf[0],sf[1]),ko(sf[2],sf[3])]
    for t in fn: Rcnt[t]["fin"]+=1
    Rcnt[ko(fn[0],fn[1])]["champ"]+=1
rows=sorted([(t,NAME[t],GOF[t],Rcnt[t]["qual"]/N_SIMS,Rcnt[t]["r16"]/N_SIMS,Rcnt[t]["qf"]/N_SIMS,
             Rcnt[t]["sf"]/N_SIMS,Rcnt[t]["fin"]/N_SIMS,Rcnt[t]["champ"]/N_SIMS) for t in GOF],key=lambda r:-r[8])
print("\nTop 8 título (FORMA+PRIOR):")
for r in rows[:8]: print(f"{r[1]:<12} campeón {r[8]*100:5.1f}%")

# equivalencias para los 16 partidos faltantes
equiv=[]
for g,h,a in REMAINING:
    lh,la=lams(h,a)
    def nearest(x):
        return sorted([(abs(ELO[u]-ELO[x]),u) for u in GOF if u!=x])[:2]
    eq_h=nearest(a); eq_a=nearest(h)  # rival equivalente
    equiv.append(dict(g=g,h=h,a=a,elo_h=ELO[h],elo_a=ELO[a],lh=lh,la=la,
        eq_for_h=[u for _,u in eq_h], eq_for_a=[u for _,u in eq_a]))

pickle.dump(dict(rows=rows,N=N_SIMS,ELO=ELO,A0=A0,D0=D0,ATT=ATT,DFN=DFN,NAME=NAME,GOF=GOF,
 acc_form=acc_form,auc_form=auc_form,acc_prior=acc_prior,auc_prior=auc_prior,equiv=equiv,KP=KP,BETA=BETA),
 open("elo_model.pkl","wb"))
print("\nOK")
