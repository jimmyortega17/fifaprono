"""
EXTENSIÓN — Simulación Monte Carlo de TODO el torneo desde el estado actual (25-jun).
- Rellena la 3.ª fecha ya jugada (A,B,C,E) en la hoja 'Mundial 2026'.
- Simula los 16 partidos de grupo restantes + llave completa (16vos→Final, con penales).
- Estima P(clasificar), P(8vos), P(4tos), P(semis), P(final), P(campeón) por selección.
"""
import numpy as np, pickle
from copy import copy
from scipy.optimize import linear_sum_assignment
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

RNG=np.random.default_rng(7); N_SIMS=6000; K_SHRINK=3.0; N_ITER=60

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

# fuerzas
def fit_strengths(matches,k=K_SHRINK,iters=N_ITER):
    teams=sorted({m[0] for m in matches}|{m[3] for m in matches})
    L=sum(m[1]+m[2] for m in matches)/(2*len(matches))
    att={t:1.0 for t in teams}; dfn={t:1.0 for t in teams}
    for _ in range(iters):
        na,nd={},{}
        for t in teams:
            gfn=gfd=gan=gad=0.0
            for h,hg,ag,a in matches:
                if t==h: gfn+=hg; gfd+=L*dfn[a]; gan+=ag; gad+=L*att[a]
                elif t==a: gfn+=ag; gfd+=L*dfn[h]; gan+=hg; gad+=L*att[h]
            na[t]=(gfn+k)/(gfd+k); nd[t]=(gan+k)/(gad+k)
        ma=np.mean(list(na.values())); md=np.mean(list(nd.values()))
        att={t:na[t]/ma for t in teams}; dfn={t:nd[t]/md for t in teams}
    return att,dfn,L
ATT,DFN,L=fit_strengths(PLAYED)
def lams(a,b): return L*ATT[a]*DFN[b], L*ATT[b]*DFN[a]

# standings actuales
base_pts={t:0 for t in GOF}; base_gf={t:0 for t in GOF}; base_ga={t:0 for t in GOF}
for h,hg,ag,a in PLAYED:
    base_gf[h]+=hg; base_ga[h]+=ag; base_gf[a]+=ag; base_ga[a]+=hg
    if hg>ag: base_pts[h]+=3
    elif ag>hg: base_pts[a]+=3
    else: base_pts[h]+=1; base_pts[a]+=1

# slots de la llave (estructura del archivo del usuario)
# P-i = (slotLocal, slotVisita).  '1X'/'2X' = 1°/2° de grupo X ; 'T'=tercero
R32=[("2A","2B"),("1C","2F"),("1E","T"),("1F","2C"),("2E","2I"),("1I","T"),("1A","T"),("1L","T"),
     ("1D","2J"),("1H","2L"),("2K","T"),("1B","2G"),("2D","2H"),("1J","T"),("1K","T"),("1G","T")]
THIRD_SLOTS=[2,5,6,7,10,13,14,15]                   # índices de R32 con tercero
THIRD_EXCL ={2:"E",5:"I",6:"A",7:"L",10:"K",13:"J",14:"K",15:"G"}  # grupo rival (no rematch)

def poisson_match(a,b,rng):
    la,lb=lams(a,b); return rng.poisson(la),rng.poisson(lb)
def knockout(a,b,rng):
    ga,gb=poisson_match(a,b,rng)
    if ga>gb: return a
    if gb>ga: return b
    pa=ATT[a]/DFN[a]; pb=ATT[b]/DFN[b]               # penales ~ proporción de poder
    return a if rng.random()<pa/(pa+pb) else b

# contadores
R={t:dict(qual=0,r16=0,qf=0,sf=0,fin=0,champ=0) for t in GOF}

for s in range(N_SIMS):
    pts=dict(base_pts); gf=dict(base_gf); ga=dict(base_ga)
    for g,h,a in REMAINING:
        hg,ag=poisson_match(h,a,RNG)
        gf[h]+=hg; ga[h]+=ag; gf[a]+=ag; ga[a]+=hg
        if hg>ag: pts[h]+=3
        elif ag>hg: pts[a]+=3
        else: pts[h]+=1; pts[a]+=1
    # standings por grupo
    pos={}; thirds=[]
    for g,ts in GROUPS.items():
        rk=sorted(ts,key=lambda t:(pts[t],gf[t]-ga[t],gf[t],RNG.random()),reverse=True)
        pos[f"1{g}"]=rk[0]; pos[f"2{g}"]=rk[1]
        t3=rk[2]; thirds.append((t3,pts[t3],gf[t3]-ga[t3],gf[t3]))
    # mejores 8 terceros
    thirds.sort(key=lambda x:(x[1],x[2],x[3],RNG.random()),reverse=True)
    q3=[t for t,_,_,_ in thirds[:8]]
    for t in [pos[f"1{g}"] for g in GROUPS]+[pos[f"2{g}"] for g in GROUPS]+q3:
        R[t]["qual"]+=1
    # asignar terceros a slots respetando exclusión de grupo (matching)
    cost=np.zeros((8,8))
    for i,t in enumerate(q3):
        for j,slot in enumerate(THIRD_SLOTS):
            cost[i,j]=1e6 if GOF[t]==THIRD_EXCL[slot] else RNG.random()
    ri,cj=linear_sum_assignment(cost)
    slot_team={}
    for i,j in zip(ri,cj): slot_team[THIRD_SLOTS[j]]=q3[i]
    # construir R32
    def resolve(slot,idx):
        return slot_team[idx] if slot=="T" else pos[slot]
    r32_teams=[(resolve(a,i),resolve(b,i)) for i,(a,b) in enumerate(R32)]
    # 16vos
    w16=[knockout(a,b,RNG) for a,b in r32_teams]
    for t in w16: R[t]["r16"]+=1
    # 8vos: (W P1,W P2),(W P3,W P4)...
    qf=[knockout(w16[i],w16[i+1],RNG) for i in range(0,16,2)]
    for t in qf: R[t]["qf"]+=1
    # 4tos
    sf=[knockout(qf[i],qf[i+1],RNG) for i in range(0,8,2)]
    for t in sf: R[t]["sf"]+=1
    # semis
    fin=[knockout(sf[0],sf[1],RNG),knockout(sf[2],sf[3],RNG)]
    for t in fin: R[t]["fin"]+=1
    champ=knockout(fin[0],fin[1],RNG); R[champ]["champ"]+=1

rows=[]
for t in GOF:
    d=R[t]
    rows.append((t,NAME[t],GOF[t],d["qual"]/N_SIMS,d["r16"]/N_SIMS,d["qf"]/N_SIMS,d["sf"]/N_SIMS,d["fin"]/N_SIMS,d["champ"]/N_SIMS))
rows.sort(key=lambda r:-r[8])
pickle.dump(dict(rows=rows,N=N_SIMS),open("tourney.pkl","wb"))
print("Top 10 candidatos al título:")
for r in rows[:10]:
    print(f"{r[1]:<12} clasif {r[3]*100:5.1f}%  8vos {r[4]*100:5.1f}%  4tos {r[5]*100:5.1f}%  semis {r[6]*100:5.1f}%  final {r[7]*100:5.1f}%  CAMPEÓN {r[8]*100:5.1f}%")

# ===================== ESCRIBIR EN EL LIBRO =====================
NAVY="0F1A33";HDR="1F3A5F";SUB="2E4A6B";GREEN="155E27";GREENF="D0EDD6";GOLD="B8860B";GOLDF="FFF2CC";ROWA="EAF0F7";WHITE="FFFFFF";DARK="0F1115"
F="Arial";thin=Side(style="thin",color="BFD0E3");bd=Border(thin,thin,thin,thin)
ctr=Alignment(horizontal="center",vertical="center");lft=Alignment(horizontal="left",vertical="center",indent=1)
def fill(h):return PatternFill("solid",fgColor=h)

wb=load_workbook("/home/claude/Mundial_2026_Full.xlsx")
ws=wb["Mundial 2026"]
def cp(dst,src):
    s=ws[src];dd=ws[dst]
    dd.font=copy(s.font);dd.fill=copy(s.fill);dd.border=copy(s.border);dd.alignment=copy(s.alignment)
def set_left(row,g1,g2):
    cp(f"B{row}","B4");cp(f"F{row}","F4")
    for col,src,val in [("C","C4",g1),("D","D4","-"),("E","E4",g2)]: cp(f"{col}{row}",src);ws[f"{col}{row}"]=val
def set_right(row,g1,g2):
    cp(f"J{row}","J4");cp(f"N{row}","N4")
    for col,src,val in [("K","K4",g1),("L","L4","-"),("M","M4",g2)]: cp(f"{col}{row}",src);ws[f"{col}{row}"]=val
# 3.ª fecha jugada
set_left(8,0,3); set_left(9,1,0)        # A: CZE 0-3 MEX ; RSA 1-0 KOR
set_right(8,2,1); set_right(9,3,1)      # B: SUI 2-1 CAN ; BIH 3-1 QAT
set_left(17,0,3); set_left(18,4,2)      # C: SCO 0-3 BRA ; MAR 4-2 HAI
set_left(26,0,2); set_left(27,2,1)      # E: CUW 0-2 CIV ; ECU 2-1 GER

# hoja de probabilidades
if "Prob. Título" in wb.sheetnames: del wb["Prob. Título"]
ws2=wb.create_sheet("Prob. Título")
ws2.sheet_view.showGridLines=False
ws2.merge_cells(start_row=1,start_column=1,end_row=1,end_column=9)
c=ws2.cell(1,1,f"PROBABILIDAD POR RONDA · {N_SIMS:,} simulaciones Monte Carlo de todo el torneo");c.fill=fill(NAVY);c.font=Font(name=F,bold=True,size=13,color=WHITE);c.alignment=ctr;ws2.row_dimensions[1].height=24
heads=["#","Cód.","Selección","Clasifica %","8vos %","4tos %","Semis %","Final %","CAMPEÓN %"]
ws2.append([]); 
for i,t in enumerate(heads,1):
    cc=ws2.cell(2,i,t);cc.fill=fill(HDR);cc.font=Font(name=F,bold=True,size=9,color=WHITE);cc.alignment=ctr;cc.border=bd
for w,col in zip([5,7,16,12,9,9,9,9,12],"ABCDEFGHI"): ws2.column_dimensions[col].width=w
for i,r in enumerate(rows,1):
    rr=2+i
    vals=[i,r[0],r[1],round(r[3]*100,1),round(r[4]*100,1),round(r[5]*100,1),round(r[6]*100,1),round(r[7]*100,1),round(r[8]*100,1)]
    for j,v in enumerate(vals,1):
        cc=ws2.cell(rr,j,v);cc.border=bd;cc.alignment=ctr if j!=3 else lft;cc.font=Font(name=F,size=9,color=DARK)
        if j==1: cc.fill=fill(SUB);cc.font=Font(name=F,bold=True,size=9,color=WHITE)
        elif j==9: cc.fill=fill(GOLDF);cc.font=Font(name=F,bold=True,size=9,color=GOLD)
        else: cc.fill=fill(ROWA if i%2 else WHITE)
ws2.freeze_panes="A3"
nr=3+len(rows)+1
ws2.merge_cells(start_row=nr,start_column=1,end_row=nr,end_column=9)
ws2.cell(nr,1,"Simulación: 16 partidos de grupo restantes + llave completa (16vos→Final) con goles ~ Poisson y penales por poder ofensivo. "
 "Asignación de terceros aproxima la tabla FIFA (respeta exclusión de grupo). Estimación probabilística, no predicción.").font=Font(name=F,size=8,italic=True,color="555555")

wb.save("/home/claude/Mundial_2026_Full.xlsx")
print("\nGuardado. Hojas:",wb.sheetnames)
