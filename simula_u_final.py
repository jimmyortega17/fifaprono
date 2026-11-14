import sys, numpy as np
sys.path.insert(0,'/home/claude/liga1')
import modelo_liga1_walkforward as M
import motivacion_liga1 as MO
rng=np.random.default_rng(7)
k=M.tidx('C',17)
PEND=[('Comerciantes Unidos','Sport Boys'),('Juan Pablo II','Dep. Garcilaso'),('UTC','Los Chankas'),('CD Moquegua','Atlético Grau'),('Cusco FC','ADT'),
('Alianza Atlético','Melgar'),('Alianza Lima','FC Cajamarca'),('Sport Huancayo','Universitario'),('Sporting Cristal','Cienciano')]
test=[(k,M.IDX[h],M.IDX[a],0,0,'C',17) for h,a in PEND]
preds,rho=M.predict_fecha(list(M.DATA),test,k+1)
mats=[]
for q,(h,a) in zip(preds,PEND):
    lh,la=MO.adjust(q['lh'],q['la'],h,a); mats.append(M.dc_matrix(lh,la,rho))
# tablas tras fecha 16
ACUM={'Alianza Lima':(69,30),'Universitario':(61,20),'Melgar':(57,18),'Dep. Garcilaso':(55,15),'Sporting Cristal':(51,15),'Cusco FC':(51,-1),'Cienciano':(50,8),
'Sport Huancayo':(50,-1),'Alianza Atlético':(49,1),'Los Chankas':(48,-12),'Sport Boys':(42,2),'Juan Pablo II':(41,-11),'Comerciantes Unidos':(37,-3),'Atlético Grau':(37,-9),
'ADT':(31,-12),'CD Moquegua':(30,-21),'FC Cajamarca':(29,-19),'UTC':(14,-20)}
CLAU={'Sport Huancayo':(34,9),'Sporting Cristal':(32,17),'Universitario':(32,11)}
N=200000
samp=[]
for Mx in mats:
    flat=Mx.ravel(); idx=rng.choice(flat.size,size=N,p=flat/flat.sum())
    samp.append((idx//Mx.shape[1], idx%Mx.shape[1]))
def sim_table(base, teams_pts):
    pass
names=list(ACUM)
pts=np.array([[ACUM[t][0] for t in names]]*N,float); gd=np.array([[ACUM[t][1] for t in names]]*N,float)
ci={t:i for i,t in enumerate(names)}
cp={t:np.full(N,CLAU[t][0],float) for t in CLAU}; cg={t:np.full(N,CLAU[t][1],float) for t in CLAU}
for (h,a),(gh,ga) in zip(PEND,samp):
    ph=np.where(gh>ga,3,np.where(gh==ga,1,0)); pa=np.where(ga>gh,3,np.where(gh==ga,1,0))
    pts[:,ci[h]]+=ph; pts[:,ci[a]]+=pa; gd[:,ci[h]]+=gh-ga; gd[:,ci[a]]+=ga-gh
    for t,p_,g_ in ((h,ph,gh-ga),(a,pa,ga-gh)):
        if t in CLAU: cp[t]+=p_; cg[t]+=g_
# campeón Clausura (puntos, luego DG, luego azar)
T=list(CLAU); score=np.stack([cp[t]*1000+cg[t]+rng.random(N)*0.01 for t in T],1)
champ=np.array(T)[score.argmax(1)]
# empates exactos en puntos y DG
tied=np.sum(np.isclose(np.sort(score,1)[:,-1],np.sort(score,1)[:,-2],atol=0.005))
print('Clausura campeón: ',{t:f"{(champ==t).mean():.1%}" for t in T}, '| empate total pts+DG:',f"{tied/N:.2%}")
# ranking acumulado (pts, DG, azar)
key=pts*10000+gd+rng.random(pts.shape)*0.01
rank=(-key).argsort(1).argsort(1)+1
h_top8=rank[:,ci['Sport Huancayo']]<=8
u_rank=rank[:,ci['Universitario']]
print('U 2.º del acumulado:',f"{(u_rank==2).mean():.1%}", '| Huancayo top-8:',f"{h_top8.mean():.1%}", '| Cristal top-8:',f"{(rank[:,ci['Sporting Cristal']]<=8).mean():.1%}")

# semifinal: U de local (supuesto: mejor posición) vs rival. avanza = gana 90' + 0.5*empate
T2=[(k,M.IDX['Universitario'],M.IDX['Sporting Cristal'],0,0,'C',17),(k,M.IDX['Universitario'],M.IDX['Sport Huancayo'],0,0,'C',17)]
pp,_=M.predict_fecha(list(M.DATA),T2,k+1)
padv={}
for q,t in zip(pp,('Sporting Cristal','Sport Huancayo')):
    P=M.probs_1x2(q['M']); padv[t]=P[0]+0.5*P[1]
    print(f"Semifinal U (local) vs {t}: gana90 {P[0]:.1%} empate {P[1]:.1%} pierde {P[2]:.1%} -> avanza {padv[t]:.1%}")
direct=(champ=='Universitario')|((champ=='Sport Huancayo')&(~h_top8))
semi_cr=(champ=='Sporting Cristal')
semi_hu=(champ=='Sport Huancayo')&h_top8
p_final=direct.mean()+semi_cr.mean()*padv['Sporting Cristal']+semi_hu.mean()*padv['Sport Huancayo']
print(f"U directo a la final: {direct.mean():.1%} (U campeón {(champ=='Universitario').mean():.1%} + Huancayo campeón fuera del top-8 {((champ=='Sport Huancayo')&(~h_top8)).mean():.1%})")
print(f"U semifinal vs Cristal: {semi_cr.mean():.1%} | vs Huancayo: {semi_hu.mean():.1%}")
print(f"U llega a la final: {p_final:.1%}  | U se queda en semifinal: {1-p_final:.1%}")
