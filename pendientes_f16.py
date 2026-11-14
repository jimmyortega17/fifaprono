import sys, numpy as np
sys.path.insert(0,'/home/claude/liga1')
import modelo_liga1_walkforward as M
from mercados_liga1 import markets
k=M.tidx('C',16)+1
train=list(M.DATA)
PEND=[('Los Chankas','Alianza Lima'),('Sport Boys','UTC'),('Dep. Garcilaso','Comerciantes Unidos'),('Universitario','CD Moquegua')]
test=[(k,M.IDX[h],M.IDX[a],0,0,'C',16) for h,a in PEND]
preds,rho=M.predict_fecha(train,test,k)
print(len(train),'partidos de entrenamiento; rho',round(rho,3))
for q in preds:
    P=markets(q['M']); Mx=q['M']; i,j=np.unravel_index(Mx.argmax(),Mx.shape)
    dc={'1X':P['1X (local o empate)'],'X2':P['X2 (visita o empate)'],'12':P['12 (sin empate)']}; kdc=max(dc,key=dc.get)
    g={n:v for n,v in P.items() if (n.startswith('Más de ') or n.startswith('Menos de ')) and '+' not in n and n!='Más de 0.5'}; b=max(g,key=g.get)
    ex=[(n,v) for n,v in P.items() if n in ('Local anota','Visita anota','Ambos anotan SI','Ambos anotan NO')]; e=max(ex,key=lambda t:t[1])
    print(f"{q['h']} - {q['a']} | esp {q['lh']+q['la']:.2f} | L/E/V {P['Local gana']:.0%}/{P['Empate']:.0%}/{P['Visita gana']:.0%} | DC {kdc} {dc[kdc]:.0%} (1X {dc['1X']:.0%}, X2 {dc['X2']:.0%}, 12 {dc['12']:.0%}) | adicional {e[0]} {e[1]:.0%} | marcador {i}-{j} {Mx[i,j]:.0%} | goles {b} {g[b]:.0%} | +0.5 {P['Más de 0.5']:.0%}")
