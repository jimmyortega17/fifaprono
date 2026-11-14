import sys, math, numpy as np
sys.path.insert(0,'/home/claude/liga1')
import modelo_liga1_walkforward as M
from mercados_liga1 import markets
k=M.tidx('C',17)
train=list(M.DATA)
PEND=[('Comerciantes Unidos','Sport Boys','Dom 15/11 1:00 p. m.'),('Juan Pablo II','Dep. Garcilaso','Dom 15/11 3:15 p. m.'),
('UTC','Los Chankas','Lun 16/11 1:00 p. m.'),('CD Moquegua','Atlético Grau','Lun 16/11 3:15 p. m.'),('Cusco FC','ADT','Lun 16/11 3:15 p. m.'),
('Alianza Atlético','Melgar','Mié 18/11 3:00 p. m.'),('Alianza Lima','FC Cajamarca','Sáb 21/11 3:30 p. m.'),('Sport Huancayo','Universitario','Dom 22/11 3:00 p. m.'),('Sporting Cristal','Cienciano','Dom 22/11 3:00 p. m.')]
test=[(k,M.IDX[h],M.IDX[a],0,0,'C',17) for h,a,_ in PEND]
preds,rho=M.predict_fecha(train,test,k+1)
print(len(train),'partidos de entrenamiento; rho',round(rho,3))
rows=[]
for q,(h,a,fh) in zip(preds,PEND):
    P=markets(q['M']); Mx=q['M']; i,j=np.unravel_index(Mx.argmax(),Mx.shape)
    dc={'1X':P['1X (local o empate)'],'X2':P['X2 (visita o empate)'],'12':P['12 (sin empate)']}; kdc=max(dc,key=dc.get)
    g={n:v for n,v in P.items() if (n.startswith('Más de ') or n.startswith('Menos de ')) and '+' not in n and n!='Más de 0.5'}; b=max(g,key=g.get)
    ex={n:P[n] for n in ('Local anota','Visita anota')}; e=max(ex,key=ex.get)
    rows.append((dc[kdc],f"{h} – {a} ({fh})",kdc,dc,e,ex[e],f"{i}-{j}",Mx[i,j],b,g[b],q['lh']+q['la'],P))
rows.sort(key=lambda t:-t[0])
for n,t in enumerate(rows,1):
    P=t[11]
    print(f"{n} {t[1]} | L/E/V {P['Local gana']:.0%}/{P['Empate']:.0%}/{P['Visita gana']:.0%} | DC {t[2]} {t[0]:.1%} (1X {t[3]['1X']:.0%} X2 {t[3]['X2']:.0%} 12 {t[3]['12']:.0%}) | {t[4]} {t[5]:.0%} | marc {t[6]} {t[7]:.0%} | {t[8]} {t[9]:.0%} | esp {t[10]:.2f}")
for n in (3,4,5):
    print(n, f"{math.prod(t[0] for t in rows[:n]):.1%}")
