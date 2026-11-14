"""
AUDITORÍA WALK-FORWARD (sin look-ahead) — Modelo 1 vs Modelo 2
Antes de cada ronda se entrena cada modelo SOLO con lo anterior y se predice a ciegas.
"""
import numpy as np
from scipy.stats import poisson, chi2

ELO0={"ESP":2171,"ARG":2113,"FRA":2063,"ENG":2042,"COL":1998,"BRA":1979,"POR":1976,"NED":1959,"CRO":1933,"ECU":1933,"NOR":1922,"GER":1910,"SUI":1897,"URU":1890,"TUR":1880,"JPN":1879,"SEN":1869,"BEL":1849,"MEX":1834,"PAR":1833,"AUT":1818,"MAR":1806,"CAN":1806,"SCO":1790,"KOR":1784,"AUS":1774,"IRN":1754,"USA":1747,"PAN":1743,"UZB":1735,"CZE":1731,"DZA":1728,"JOR":1691,"SWE":1660,"EGY":1660,"COD":1639,"CIV":1637,"TUN":1614,"KSA":1592,"NZL":1586,"IRQ":1583,"BIH":1571,"CPV":1561,"HAI":1542,"RSA":1529,"GHA":1509,"CUW":1467,"QAT":1427}
def g(l): return [(h,a,b,c) for h,a,b,c in l]
J1=g([("MEX",2,0,"RSA"),("KOR",2,1,"CZE"),("CAN",1,1,"BIH"),("QAT",1,1,"SUI"),("BRA",1,1,"MAR"),("HAI",0,1,"SCO"),("USA",4,1,"PAR"),("AUS",2,0,"TUR"),("GER",7,1,"CUW"),("CIV",1,0,"ECU"),("NED",2,2,"JPN"),("SWE",5,1,"TUN"),("BEL",1,1,"EGY"),("IRN",2,2,"NZL"),("ESP",0,0,"CPV"),("KSA",1,1,"URU"),("FRA",3,1,"SEN"),("IRQ",1,4,"NOR"),("ARG",3,0,"DZA"),("AUT",3,1,"JOR"),("POR",1,1,"COD"),("UZB",1,3,"COL"),("ENG",4,2,"CRO"),("GHA",1,0,"PAN")])
J2=g([("CZE",1,1,"RSA"),("MEX",1,0,"KOR"),("SUI",4,1,"BIH"),("CAN",6,0,"QAT"),("SCO",0,1,"MAR"),("BRA",3,0,"HAI"),("USA",2,0,"AUS"),("TUR",0,1,"PAR"),("GER",2,1,"CIV"),("ECU",0,0,"CUW"),("NED",5,1,"SWE"),("TUN",0,4,"JPN"),("BEL",0,0,"IRN"),("NZL",1,3,"EGY"),("ESP",4,0,"KSA"),("URU",2,2,"CPV"),("FRA",3,0,"IRQ"),("NOR",3,2,"SEN"),("ARG",2,0,"AUT"),("JOR",1,2,"DZA"),("POR",5,0,"UZB"),("COL",1,0,"COD"),("ENG",0,0,"GHA"),("CRO",1,0,"PAN")])
J3=g([("CZE",0,3,"MEX"),("RSA",1,0,"KOR"),("SUI",2,1,"CAN"),("BIH",3,1,"QAT"),("SCO",0,3,"BRA"),("MAR",4,2,"HAI"),("TUR",3,2,"USA"),("PAR",0,0,"AUS"),("CUW",0,2,"CIV"),("ECU",2,1,"GER"),("TUN",1,3,"NED"),("JPN",1,1,"SWE"),("EGY",1,1,"IRN"),("NZL",1,5,"BEL"),("CPV",0,0,"KSA"),("URU",0,1,"ESP"),("NOR",1,4,"FRA"),("SEN",5,0,"IRQ"),("DZA",3,3,"AUT"),("JOR",1,3,"ARG"),("COL",0,0,"POR"),("COD",3,1,"UZB"),("PAN",0,2,"ENG"),("CRO",2,1,"GHA")])
R32=g([("RSA",0,1,"CAN"),("BRA",2,1,"JPN"),("GER",1,1,"PAR"),("NED",1,1,"MAR"),("CIV",1,2,"NOR"),("FRA",3,0,"SWE"),("MEX",2,0,"ECU"),("ENG",2,1,"COD"),("BEL",3,2,"SEN"),("USA",2,0,"BIH"),("ESP",3,0,"AUT"),("POR",2,1,"CRO"),("SUI",2,0,"DZA"),("AUS",1,1,"EGY"),("ARG",3,2,"CPV"),("COL",1,0,"GHA")])
R16=g([("CAN",0,3,"MAR"),("PAR",0,1,"FRA"),("BRA",1,2,"NOR"),("MEX",2,3,"ENG"),("POR",0,1,"ESP"),("USA",1,4,"BEL"),("ARG",3,2,"EGY"),("SUI",0,0,"COL")])
QF=g([("FRA",2,0,"MAR")])
ROUNDS=[("J1",J1,1),("J2",J2,2),("J3",J3,3),("R32",R32,0),("R16",R16,0),("QF",QF,0)]
PEN={("GER","PAR"):"PAR",("NED","MAR"):"MAR",("AUS","EGY"):"EGY",("SUI","COL"):"SUI"}

def lab(h,a): return "H" if h>a else ("A" if a>h else "D")
def tau(x,y,l1,l2,r):
    if x==0 and y==0:return 1-l1*l2*r
    if x==0 and y==1:return 1+l1*r
    if x==1 and y==0:return 1+l2*r
    if x==1 and y==1:return 1-r
    return 1.0
def mat(l1,l2,rho,mg=9):
    px=poisson.pmf(np.arange(mg+1),l1);py=poisson.pmf(np.arange(mg+1),l2);M=np.outer(px,py)
    for x in range(2):
        for y in range(2):M[x,y]*=tau(x,y,l1,l2,rho)
    return M/M.sum()

# ---- MODELO 1: Poisson anclado en prior Elo (att/def con contracción) ----
def fit_attdef(train):
    teams=sorted(ELO0)
    if not train: 
        eb=np.mean([ELO0[t] for t in teams]);A0={t:np.exp(0.002*(ELO0[t]-eb))**.5 for t in teams};D0={t:np.exp(0.002*(ELO0[t]-eb))**-.5 for t in teams}
        mA=np.mean(list(A0.values()));mD=np.mean(list(D0.values()))
        return {t:A0[t]/mA for t in teams},{t:D0[t]/mD for t in teams},1.4
    L=sum(a+b for h,a,b,c in train)/(2*len(train));eb=np.mean([ELO0[t] for t in teams])
    A0={t:np.exp(0.002*(ELO0[t]-eb))**.5 for t in teams};D0={t:np.exp(0.002*(ELO0[t]-eb))**-.5 for t in teams}
    mA=np.mean(list(A0.values()));mD=np.mean(list(D0.values()));A0={t:A0[t]/mA for t in teams};D0={t:D0[t]/mD for t in teams}
    att={t:1. for t in teams};dfn={t:1. for t in teams}
    for _ in range(60):
        na,nd={},{}
        for t in teams:
            fn=fd=an=ad=0.
            for h,g1,g2,a in train:
                if t==h:fn+=g1;fd+=L*dfn[a];an+=g2;ad+=L*att[a]
                elif t==a:fn+=g2;fd+=L*dfn[h];an+=g1;ad+=L*att[h]
            na[t]=(fn+4*L*A0[t])/(fd+4*L);nd[t]=(an+4*L*D0[t])/(ad+4*L)
        mA=np.mean(list(na.values()));mD=np.mean(list(nd.values()));att={t:na[t]/mA for t in teams};dfn={t:nd[t]/mD for t in teams}
    return att,dfn,L
def model1(train,h,a,md):
    att,dfn,L=fit_attdef(train);l1=L*att[h]*dfn[a];l2=L*att[a]*dfn[h]
    M=mat(l1,l2,0.0);return M,l1,l2

# ---- MODELO 2: Dixon-Coles + Elo dinámico + boost J3 + goleada ----
def dyn_elo(train):
    e=dict(ELO0)
    for h,g1,g2,a in train:
        p=1/(1+10**((e[a]-e[h])/400));s=1 if g1>g2 else(0.5 if g1==g2 else 0);k=40*(1+0.1*max(abs(g1-g2)-2,0));d=k*(s-p);e[h]+=d;e[a]-=d
    return e
def model2(train,h,a,md):
    e=dyn_elo(train);pw=1/(1+10**((e[a]-e[h])/400));base=1.3942*(1.15 if md==3 else 1.0)
    l1=base*(0.5+pw);l2=base*(1.5-pw);M=mat(l1,l2,-0.13);return M,l1,l2
def goleada_score(M,l1,l2):
    b=list(np.unravel_index(M.argmax(),M.shape))
    if l1>=1.75 and (l1-l2)>0.8:b[0]=max(b[0],round(l1))
    if l2>=1.75 and (l2-l1)>0.8:b[1]=max(b[1],round(l2))
    return b[0],b[1]

# ---- WALK-FORWARD ----
cls=["H","D","A"]
def audit(model,goleada=False):
    hist=[];rec=[]
    for name,rnd,md in ROUNDS:
        train=list(hist)
        for h,g1,g2,a in rnd:
            M,l1,l2=model(train,h,a,md)
            p=[np.tril(M,-1).sum(),np.trace(M),np.triu(M,1).sum()]
            if goleada: sc=goleada_score(M,l1,l2)
            else: sc=np.unravel_index(M.argmax(),M.shape)
            rec.append(dict(h=h,a=a,g1=g1,g2=g2,p=p,sc=(int(sc[0]),int(sc[1])),l1=l1,l2=l2,
                            real=lab(g1,g2),ko=(name in("R32","R16","QF")),
                            adv=(h if g1>g2 else (a if g2>g1 else PEN.get((h,a))))))
        hist+=list(rnd)
    return rec

r1=audit(model1,goleada=False)
r2=audit(model2,goleada=True)

def metrics(rec):
    n=len(rec);acc=exact=0;ll=brier=mae=rmse=0;ko_ok=ko_n=0;probs=[];corr=[]
    for r in rec:
        pred=cls[int(np.argmax(r["p"]))]
        ok=(pred==r["real"]);acc+=ok;corr.append(ok)
        exact+=(r["sc"]==(r["g1"],r["g2"]))
        pr=r["p"][cls.index(r["real"])];probs.append(pr)
        ll+=-np.log(max(pr,1e-9))
        oh=[1 if c==r["real"] else 0 for c in cls];brier+=sum((r["p"][i]-oh[i])**2 for i in range(3))
        mae+=abs(r["l1"]-r["g1"])+abs(r["l2"]-r["g2"]);rmse+=(r["l1"]-r["g1"])**2+(r["l2"]-r["g2"])**2
        if r["ko"] and r["adv"]:
            ko_n+=1;ko_ok+=(( r["h"] if r["p"][0]>=r["p"][2] else r["a"])==r["adv"])
    # ECE (10 bins sobre prob máxima)
    conf=[max(r["p"]) for r in rec];ece=0
    for lo in np.arange(0,1,0.1):
        idx=[i for i in range(n) if lo<=conf[i]<lo+0.1]
        if idx:
            a_=np.mean([corr[i] for i in idx]);c_=np.mean([conf[i] for i in idx]);ece+=len(idx)/n*abs(a_-c_)
    return dict(n=n,acc=acc/n,exact=exact/n,ll=ll/n,brier=brier/n,mae=mae/(2*n),rmse=np.sqrt(rmse/(2*n)),
                ko=ko_ok/ko_n if ko_n else 0,ko_n=ko_n,ece=ece),corr

m1,c1=metrics(r1);m2,c2=metrics(r2)
# McNemar
b=sum(1 for i in range(len(c1)) if c1[i] and not c2[i]);c=sum(1 for i in range(len(c1)) if c2[i] and not c1[i])
stat=(abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0;pval=1-chi2.cdf(stat,1)

print(f"{'Métrica':<28}{'Modelo 1':>12}{'Modelo 2':>12}")
print(f"{'N partidos':<28}{m1['n']:>12}{m2['n']:>12}")
print(f"{'Acierto 1X2':<28}{m1['acc']*100:>11.1f}%{m2['acc']*100:>11.1f}%")
print(f"{'Acierto avance (KO, n='+str(m1['ko_n'])+')':<28}{m1['ko']*100:>11.1f}%{m2['ko']*100:>11.1f}%")
print(f"{'Marcador exacto':<28}{m1['exact']*100:>11.1f}%{m2['exact']*100:>11.1f}%")
print(f"{'MAE goles':<28}{m1['mae']:>12.3f}{m2['mae']:>12.3f}")
print(f"{'RMSE goles':<28}{m1['rmse']:>12.3f}{m2['rmse']:>12.3f}")
print(f"{'Log Loss':<28}{m1['ll']:>12.3f}{m2['ll']:>12.3f}")
print(f"{'Brier Score':<28}{m1['brier']:>12.3f}{m2['brier']:>12.3f}")
print(f"{'Calibration Error (ECE)':<28}{m1['ece']:>12.3f}{m2['ece']:>12.3f}")
print(f"\nMcNemar: b(M1✓,M2✗)={b}  c(M1✗,M2✓)={c}  chi2={stat:.2f}  p={pval:.3f}")
a1=sum(c1);a2=sum(c2);N=len(c1)
print(f"\nModelo 1 acertó {a1}/{N} 1X2 ({a1/N*100:.1f}%)")
print(f"Modelo 2 acertó {a2}/{N} 1X2 ({a2/N*100:.1f}%)")
print("Diferencia significativa (p<0.05):", "SÍ" if pval<0.05 else "NO")
