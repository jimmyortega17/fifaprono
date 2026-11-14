"""
================================================================================
 MODELO DE PRONÓSTICO — COPA MUNDIAL DE LA FIFA 2026  (versión para publicación)
================================================================================
 Modelo Poisson–Dixon-Coles anclado en ratings Elo dinámicos, con:
   (1) Elo DINÁMICO actualizado partido a partido (K amplificado por goleada).
   (2) Modelo de goles Poisson con corrección de DIXON-COLES (marcadores bajos).
   (3) Medias de goles (lambda) derivadas de la diferencia de Elo.
   (4) CORRECCIÓN DE GOLEADA: el marcador modal deja de subestimar al favorito
       claro (evita el sesgo "1-1 en todos lados").
   (5) VENTAJA DE ANFITRIÓN parametrizable (+HB Elo a EE.UU./México/Canadá).
   (6) Resolución de eliminatoria en TRES FASES: 90' -> prórroga (120') -> penales.
   (7) Validación WALK-FORWARD (prospectiva) por jornada + acierto en eliminación.
   (8) Simulación de Monte Carlo para probabilidad de título.

 Métricas de validación (fuera de muestra): exactitud 1X2, AUC (OvR), log-loss.
 Requisitos: numpy, scipy, scikit-learn (solo para las métricas de validación).
================================================================================
"""
import numpy as np
from scipy.stats import poisson
from collections import Counter

# =============================================================================
# 1. RATINGS ELO PRE-TORNEO (prior)
# =============================================================================
ELO0 = {
    "ESP":2171,"ARG":2113,"FRA":2063,"ENG":2042,"COL":1998,"BRA":1979,"POR":1976,
    "NED":1959,"CRO":1933,"ECU":1933,"NOR":1922,"GER":1910,"SUI":1897,"URU":1890,
    "TUR":1880,"JPN":1879,"SEN":1869,"BEL":1849,"MEX":1834,"PAR":1833,"AUT":1818,
    "MAR":1806,"CAN":1806,"SCO":1790,"KOR":1784,"AUS":1774,"IRN":1754,"USA":1747,
    "PAN":1743,"UZB":1735,"CZE":1731,"DZA":1728,"JOR":1691,"SWE":1660,"EGY":1660,
    "COD":1639,"CIV":1637,"TUN":1614,"KSA":1592,"NZL":1586,"IRQ":1583,"BIH":1571,
    "CPV":1561,"HAI":1542,"RSA":1529,"GHA":1509,"CUW":1467,"QAT":1427,
}
NOMBRE = {"ESP":"España","ARG":"Argentina","FRA":"Francia","ENG":"Inglaterra",
    "COL":"Colombia","BRA":"Brasil","POR":"Portugal","NED":"P. Bajos","CRO":"Croacia",
    "ECU":"Ecuador","NOR":"Noruega","GER":"Alemania","SUI":"Suiza","BEL":"Bélgica",
    "MEX":"México","PAR":"Paraguay","AUT":"Austria","MAR":"Marruecos","CAN":"Canadá",
    "AUS":"Australia","USA":"EE.UU.","DZA":"Argelia","EGY":"Egipto","CPV":"Cabo Verde",
    "GHA":"Ghana","BIH":"Bosnia","SEN":"Senegal","JPN":"Japón","SWE":"Suecia","SUI":"Suiza"}

# =============================================================================
# 2. PARÁMETROS DEL MODELO (todos en un solo lugar, para reproducibilidad)
# =============================================================================
BASE_LAMBDA = 1.3942      # tasa base de goles por equipo
RHO         = -0.13       # dependencia de Dixon-Coles (marcadores bajos)
K_ELO       = 40          # factor de aprendizaje del Elo dinámico
HOST        = {"USA", "MEX", "CAN"}   # anfitriones 2026
HOST_BONUS  = 35          # Elo extra por localía (calibrado; 0 = desactivar)
GOLEADA_MIN = 1.75        # umbral de lambda para activar corrección de goleada
GOLEADA_GAP = 0.80        # ventaja mínima de lambda para activar la corrección
ET_FACTOR   = 1/3         # prórroga: 30' equivalen a 1/3 del ritmo de goles
PEN_DAMP    = 2000        # amortiguación del Elo en penales (tanda ~ azar)
PEN_CONV_F  = 0.77        # conversión del favorito en la tanda
PEN_CONV_U  = 0.74        # conversión del no-favorito
N_MC        = 30000       # simulaciones de Monte Carlo

# =============================================================================
# 3. PARTIDOS JUGADOS  ->  (local, gl, gv, visita, fase)
#    fase: "J1","J2","J3" (grupos, para walk-forward), "R32","R16" (eliminación)
# =============================================================================
def _g(lst, fase): return [(h,a,b,c,fase) for (h,a,b,c) in lst]

GRUPO_J1 = _g([("MEX",2,0,"RSA"),("KOR",2,1,"CZE"),("CAN",1,1,"BIH"),("QAT",1,1,"SUI"),
    ("BRA",1,1,"MAR"),("HAI",0,1,"SCO"),("USA",4,1,"PAR"),("AUS",2,0,"TUR"),
    ("GER",7,1,"CUW"),("CIV",1,0,"ECU"),("NED",2,2,"JPN"),("SWE",5,1,"TUN"),
    ("BEL",1,1,"EGY"),("IRN",2,2,"NZL"),("ESP",0,0,"CPV"),("KSA",1,1,"URU"),
    ("FRA",3,1,"SEN"),("IRQ",1,4,"NOR"),("ARG",3,0,"DZA"),("AUT",3,1,"JOR"),
    ("POR",1,1,"COD"),("UZB",1,3,"COL"),("ENG",4,2,"CRO"),("GHA",1,0,"PAN")], "J1")
GRUPO_J2 = _g([("CZE",1,1,"RSA"),("MEX",1,0,"KOR"),("SUI",4,1,"BIH"),("CAN",6,0,"QAT"),
    ("SCO",0,1,"MAR"),("BRA",3,0,"HAI"),("USA",2,0,"AUS"),("TUR",0,1,"PAR"),
    ("GER",2,1,"CIV"),("ECU",0,0,"CUW"),("NED",5,1,"SWE"),("TUN",0,4,"JPN"),
    ("BEL",0,0,"IRN"),("NZL",1,3,"EGY"),("ESP",4,0,"KSA"),("URU",2,2,"CPV"),
    ("FRA",3,0,"IRQ"),("NOR",3,2,"SEN"),("ARG",2,0,"AUT"),("JOR",1,2,"DZA"),
    ("POR",5,0,"UZB"),("COL",1,0,"COD"),("ENG",0,0,"GHA"),("CRO",1,0,"PAN")], "J2")
GRUPO_J3 = _g([("CZE",0,3,"MEX"),("RSA",1,0,"KOR"),("SUI",2,1,"CAN"),("BIH",3,1,"QAT"),
    ("SCO",0,3,"BRA"),("MAR",4,2,"HAI"),("TUR",3,2,"USA"),("PAR",0,0,"AUS"),
    ("CUW",0,2,"CIV"),("ECU",2,1,"GER"),("TUN",1,3,"NED"),("JPN",1,1,"SWE"),
    ("EGY",1,1,"IRN"),("NZL",1,5,"BEL"),("CPV",0,0,"KSA"),("URU",0,1,"ESP"),
    ("NOR",1,4,"FRA"),("SEN",5,0,"IRQ"),("DZA",3,3,"AUT"),("JOR",1,3,"ARG"),
    ("COL",0,0,"POR"),("COD",3,1,"UZB"),("PAN",0,2,"ENG"),("CRO",2,1,"GHA")], "J3")
DIECISEISAVOS = _g([("RSA",0,1,"CAN"),("BRA",2,1,"JPN"),("GER",1,1,"PAR"),("NED",1,1,"MAR"),
    ("CIV",1,2,"NOR"),("FRA",3,0,"SWE"),("MEX",2,0,"ECU"),("ENG",2,1,"COD"),
    ("BEL",3,2,"SEN"),("USA",2,0,"BIH"),("ESP",3,0,"AUT"),("POR",2,1,"CRO"),
    ("SUI",2,0,"DZA"),("AUS",1,1,"EGY"),("ARG",3,2,"CPV"),("COL",1,0,"GHA")], "R32")
OCTAVOS = _g([("CAN",0,3,"MAR"),("PAR",0,1,"FRA"),("BRA",1,2,"NOR"),("MEX",2,3,"ENG"),
    ("POR",0,1,"ESP"),("USA",1,4,"BEL"),("ARG",3,2,"EGY"),("SUI",0,0,"COL")], "R16")
# Ganador por penales cuando el 90'+120' terminó empatado (para el bracket real)
PEN_WINNER = {("GER","PAR"):"PAR", ("NED","MAR"):"MAR", ("AUS","EGY"):"EGY", ("SUI","COL"):"SUI"}

PLAYED = GRUPO_J1 + GRUPO_J2 + GRUPO_J3 + DIECISEISAVOS + OCTAVOS

# =============================================================================
# 4. ELO DINÁMICO
# =============================================================================
def build_dynamic_elo(matches, base=ELO0):
    """Actualiza el Elo tras cada partido. K se amplifica en goleadas (|dif|>2)."""
    elo = dict(base)
    for h, hg, ag, a, _ in matches:
        exp_h = 1.0 / (1.0 + 10 ** ((elo[a] - elo[h]) / 400))
        s_h   = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k     = K_ELO * (1 + 0.1 * max(abs(hg - ag) - 2, 0))
        delta = k * (s_h - exp_h)
        elo[h] += delta; elo[a] -= delta
    return elo

# =============================================================================
# 5. DIXON-COLES + MEDIAS DE GOLES + CORRECCIÓN DE GOLEADA
# =============================================================================
def _tau(x, y, la, mu, rho):
    if x == 0 and y == 0: return 1 - la * mu * rho
    if x == 0 and y == 1: return 1 + la * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def dc_matrix(la, mu, rho=RHO, maxg=9):
    """Matriz conjunta de marcadores con corrección de Dixon-Coles."""
    px = poisson.pmf(np.arange(maxg + 1), la)
    py = poisson.pmf(np.arange(maxg + 1), mu)
    M = np.outer(px, py)
    for x in range(2):
        for y in range(2):
            M[x, y] *= _tau(x, y, la, mu, rho)
    return M / M.sum()

def lambdas(elo, a, b):
    """Medias de goles desde la diferencia de Elo (con bono de anfitrión)."""
    ea = elo[a] + (HOST_BONUS if a in HOST else 0)
    eb = elo[b] + (HOST_BONUS if b in HOST else 0)
    pw = 1.0 / (1.0 + 10 ** ((eb - ea) / 400))
    return BASE_LAMBDA * (0.5 + pw), BASE_LAMBDA * (1.5 - pw), ea, eb

def score_goleada(la, mu):
    """
    Marcador modal con CORRECCIÓN DE GOLEADA: si un equipo es claramente
    superior (lambda >= GOLEADA_MIN y ventaja > GOLEADA_GAP), su marcador se
    eleva a round(lambda) para no subestimar al favorito (evita el '1-1' crónico).
    """
    M = dc_matrix(la, mu)
    best = list(np.unravel_index(M.argmax(), M.shape))
    if la >= GOLEADA_MIN and (la - mu) > GOLEADA_GAP: best[0] = max(best[0], round(la))
    if mu >= GOLEADA_MIN and (mu - la) > GOLEADA_GAP: best[1] = max(best[1], round(mu))
    return best[0], best[1]

# =============================================================================
# 6. RESOLUCIÓN EN TRES FASES (90' -> prórroga -> penales)
# =============================================================================
def penalty_shootout(rng, pc_fav=PEN_CONV_F, pc_und=PEN_CONV_U):
    """Tanda: 5 tiros + muerte súbita. Devuelve (goles_fav, goles_und)."""
    a = b = 0
    for _ in range(5):
        a += rng.random() < pc_fav; b += rng.random() < pc_und
    while a == b:
        a += rng.random() < pc_fav; b += rng.random() < pc_und
    return a, b

def resolve_ko(elo, a, b, rng=None):
    """
    Devuelve dict con: favorito, prob. de avance, marcador 90', probabilidad de
    prórroga y de penales, y (si rng) el marcador modal de la tanda.
    """
    la, mu, ea, eb = lambdas(elo, a, b)
    M  = dc_matrix(la, mu)
    p1, pX, p2 = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    Me = dc_matrix(la * ET_FACTOR, mu * ET_FACTOR)          # prórroga (30')
    e1, eX, e2 = np.tril(Me, -1).sum(), np.trace(Me), np.triu(Me, 1).sum()
    ppa = 1.0 / (1.0 + 10 ** ((eb - ea) / PEN_DAMP))         # penales ~ azar
    adv_a = p1 + pX * (e1 + eX * ppa)
    adv_b = p2 + pX * (e2 + eX * (1 - ppa))
    fav = a if adv_a >= adv_b else b
    s90 = score_goleada(la, mu)
    pen = None
    if rng is not None:
        pen = Counter(penalty_shootout(rng) for _ in range(4000)).most_common(1)[0][0]
    return dict(fav=fav, p_adv=max(adv_a, adv_b), sc90=s90,
                p_pro=pX, p_pen=pX * eX, pen=pen)

def simulate_ko(elo, a, b, rng):
    """Simula UN partido de eliminación (tres fases). Devuelve el ganador."""
    la, mu, ea, eb = lambdas(elo, a, b)
    ga, gb = rng.poisson(la), rng.poisson(mu)
    if ga > gb: return a
    if gb > ga: return b
    ga2, gb2 = rng.poisson(la * ET_FACTOR), rng.poisson(mu * ET_FACTOR)
    if ga2 > gb2: return a
    if gb2 > ga2: return b
    return a if rng.random() < 1.0/(1.0+10**((eb-ea)/PEN_DAMP)) else b

# =============================================================================
# 7. VALIDACIÓN
# =============================================================================
def _lab(hg, ag): return "H" if hg > ag else ("A" if ag > hg else "D")

def backtest_walkforward():
    """
    Validación PROSPECTIVA de la fase de grupos: predice cada jornada entrenando
    el Elo solo con las jornadas anteriores. Devuelve acierto/AUC/logloss por ronda.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    cls = ["H", "D", "A"]
    jornadas = {"J1": GRUPO_J1, "J2": GRUPO_J2, "J3": GRUPO_J3}
    orden = ["J1", "J2", "J3"]; out = {}
    for i, jname in enumerate(orden):
        train = sum((jornadas[orden[j]] for j in range(i)), [])
        elo = build_dynamic_elo(train) if train else dict(ELO0)
        P, Y = [], []
        for h, hg, ag, a, _ in jornadas[jname]:
            la, mu, _, _ = lambdas(elo, h, a)
            M = dc_matrix(la, mu)
            P.append([np.tril(M,-1).sum(), np.trace(M), np.triu(M,1).sum()])
            Y.append(_lab(hg, ag))
        P, Y = np.array(P), np.array(Y); Yb = label_binarize(Y, classes=cls)
        acc = np.mean([cls[k] for k in P.argmax(1)] == Y)
        auc = roc_auc_score(Yb, P, average="macro", multi_class="ovr")
        ll  = -np.mean([np.log(max(P[k, cls.index(Y[k])], 1e-9)) for k in range(len(Y))])
        out[jname] = (len(Y), acc, auc, ll)
    return out

def backtest_knockout():
    """Acierto en avance de la fase de eliminación (walk-forward cronológico)."""
    elo = build_dynamic_elo(GRUPO_J1 + GRUPO_J2 + GRUPO_J3)
    ko = DIECISEISAVOS + OCTAVOS
    ok = 0; tot = 0
    for h, hg, ag, a, _ in ko:
        r = resolve_ko(elo, h, a)
        real = h if hg > ag else (a if ag > hg else PEN_WINNER.get((h, a), None))
        if real is not None:
            tot += 1; ok += (r["fav"] == real)
        # actualizar Elo con el resultado
        exp_h = 1/(1+10**((elo[a]-elo[h])/400)); s = 1.0 if hg>ag else (0.5 if hg==ag else 0)
        k = K_ELO*(1+0.1*max(abs(hg-ag)-2,0)); d = k*(s-exp_h); elo[h]+=d; elo[a]-=d
    return ok, tot

# =============================================================================
# 8. ESTADÍSTICAS DE EQUIPO (goles a favor/en contra, invicto)
# =============================================================================
def team_stats(teams):
    st = {t: dict(gf=0, gc=0, losses=0) for t in teams}
    for h, hg, ag, a, _ in PLAYED:
        for t, gf, gc in [(h, hg, ag), (a, ag, hg)]:
            if t in st:
                st[t]["gf"] += gf; st[t]["gc"] += gc
                if gf < gc: st[t]["losses"] += 1
    return st

# =============================================================================
# 9. BRACKET (árbol oficial 2026) — cuartos ya definidos
# =============================================================================
CUARTOS_MATCHES = {97:("FRA","MAR"), 98:("ESP","BEL"), 99:("NOR","ENG"), 100:("ARG","SUI")}
SEMIS = [(101, 97, 98), (102, 99, 100)]   # W101 vs W102 = FINAL

def project_bracket(elo):
    w = {}
    for m, (a, b) in CUARTOS_MATCHES.items():
        w[m] = resolve_ko(elo, a, b)["fav"]
    for m, x, y in SEMIS:
        w[m] = resolve_ko(elo, w[x], w[y])["fav"]
    champ = resolve_ko(elo, w[101], w[102])["fav"]
    return w, champ

def monte_carlo(elo, n=N_MC, seed=3):
    rng = np.random.default_rng(seed)
    ch = {t: 0 for t in elo}
    for _ in range(n):
        w = {m: simulate_ko(elo, a, b, rng) for m, (a, b) in CUARTOS_MATCHES.items()}
        for m, x, y in SEMIS: w[m] = simulate_ko(elo, w[x], w[y], rng)
        ch[simulate_ko(elo, w[101], w[102], rng)] += 1
    return {t: c / n for t, c in ch.items()}

# =============================================================================
# 10. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    elo = build_dynamic_elo(PLAYED)
    rng = np.random.default_rng(7)

    print("### VALIDACIÓN WALK-FORWARD (fase de grupos) ###")
    for j, (n, acc, auc, ll) in backtest_walkforward().items():
        print(f"  {j}: n={n}  acierto={acc*100:.1f}%  AUC={auc:.3f}  logloss={ll:.3f}")
    ok, tot = backtest_knockout()
    print(f"  Eliminación (avance): {ok}/{tot} = {ok/tot*100:.1f}%")

    print("\n### CUARTOS (90' / prórroga / penales) ###")
    for m, (a, b) in CUARTOS_MATCHES.items():
        r = resolve_ko(elo, a, b, rng)
        s = r["sc90"]
        print(f"  {NOMBRE[a]} - {NOMBRE[b]:<11} {s[0]}-{s[1]}  "
              f"(pen {r['p_pen']*100:.0f}%)  -> {NOMBRE[r['fav']]} {r['p_adv']*100:.0f}%")

    print("\n### CUADRO PROYECTADO ###")
    w, champ = project_bracket(elo)
    for m, x, y in SEMIS:
        print(f"  Semi: {NOMBRE[w[x]]} - {NOMBRE[w[y]]} -> {NOMBRE[w[m]]}")
    print(f"  FINAL: {NOMBRE[w[101]]} - {NOMBRE[w[102]]} -> CAMPEÓN {NOMBRE[champ]}")

    print("\n### PROBABILIDAD DE TÍTULO (Monte Carlo) ###")
    for t, p in sorted(monte_carlo(elo).items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {NOMBRE.get(t,t):<12} {p*100:.1f}%")

    print("\n### ESTADÍSTICAS (cuartofinalistas) ###")
    qf = [a for pair in CUARTOS_MATCHES.values() for a in pair]
    st = team_stats(qf)
    for t in sorted(qf, key=lambda x: st[x]["gc"]):
        inv = "invicto" if st[t]["losses"] == 0 else f"{st[t]['losses']} derrota(s)"
        print(f"  {NOMBRE[t]:<11} GF={st[t]['gf']:2d}  GC={st[t]['gc']:2d}  {inv}")
