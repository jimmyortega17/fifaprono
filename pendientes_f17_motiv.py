import sys, math, numpy as np
sys.path.insert(0,'/home/claude/liga1')
import modelo_liga1_walkforward as M
from mercados_liga1 import markets
import motivacion_liga1 as MO
k=M.tidx('C',17)
PEND=[('Comerciantes Unidos','Sport Boys','Dom 15/11 1:00 p. m.'),('Juan Pablo II','Dep. Garcilaso','Dom 15/11 3:15 p. m.'),
('UTC','Los Chankas','Lun 16/11 1:00 p. m.'),('CD Moquegua','Atlético Grau','Lun 16/11 3:15 p. m.'),('Cusco FC','ADT','Lun 16/11 3:15 p. m.'),
('Alianza Atlético','Melgar','Mié 18/11 3:00 p. m.'),('Alianza Lima','FC Cajamarca','Sáb 21/11 3:30 p. m.'),('Sport Huancayo','Universitario','Dom 22/11 3:00 p. m.'),('Sporting Cristal','Cienciano','Dom 22/11 3:00 p. m.')]
test=[(k,M.IDX[h],M.IDX[a],0,0,'C',17) for h,a,_ in PEND]
preds,rho=M.predict_fecha(list(M.DATA),test,k+1)
rows=[]
for q,(h,a,fh) in zip(preds,PEND):
    lh,la=MO.adjust(q['lh'],q['la'],h,a)
    Mx=M.dc_matrix(lh,la,rho); P=markets(Mx); P0=markets(q['M'])
    i,j=np.unravel_index(Mx.argmax(),Mx.shape)
    dc={'1X':P['1X (local o empate)'],'X2':P['X2 (visita o empate)'],'12':P['12 (sin empate)']}; kdc=max(dc,key=dc.get)
    g={n:v for n,v in P.items() if (n.startswith('Más de ') or n.startswith('Menos de ')) and '+' not in n and n!='Más de 0.5'}; b=max(g,key=g.get)
    ex={n:P[n] for n in ('Local anota','Visita anota')}; e=max(ex,key=ex.get)
    rows.append(dict(dcv=dc[kdc],txt=f"{h} – {a} ({fh})",dc=kdc,dcd=dc,e=e,ev=ex[e],marc=f"{i}-{j}",pm=Mx[i,j],g=b,gv=g[b],esp=lh+la,
        mh=MO.m(h),ma=MO.m(a),L=P['Local gana'],E=P['Empate'],V=P['Visita gana'],L0=P0['Local gana'],V0=P0['Visita gana'],dc0=max(P0['1X (local o empate)'],P0['X2 (visita o empate)'],P0['12 (sin empate)'])))
rows.sort(key=lambda r:-r['dcv'])
for n,r in enumerate(rows,1):
    print(f"{n} {r['txt']} | m {r['mh']}/{r['ma']} | L/E/V {r['L']:.0%}/{r['E']:.0%}/{r['V']:.0%} (antes L {r['L0']:.0%} V {r['V0']:.0%}) | DC {r['dc']} {r['dcv']:.1%} [1X {r['dcd']['1X']:.0%} X2 {r['dcd']['X2']:.0%} 12 {r['dcd']['12']:.0%}] (antes mejor {r['dc0']:.1%}) | {r['e']} {r['ev']:.0%} | marc {r['marc']} {r['pm']:.0%} | {r['g']} {r['gv']:.0%} | esp {r['esp']:.2f}")
for n in (3,4,5): print(n, f"{math.prod(r['dcv'] for r in rows[:n]):.1%}")
