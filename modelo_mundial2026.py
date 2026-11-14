"""
==============================================================================
 MODELO DE PRONÓSTICO — COPA MUNDIAL FIFA 2026
 Versión al día de hoy (fase de octavos de final)
==============================================================================
 Componentes:
   1. Ratings Elo pre-torneo como prior.
   2. Elo DINÁMICO: se actualiza tras cada partido (K=40, amplificado por goleada).
   3. Modelo de goles Poisson con corrección de Dixon-Coles (marcadores bajos).
   4. Medias de goles (lambda) derivadas de la diferencia de Elo.
   5. Ventaja de ANFITRIÓN (+35 Elo a EE.UU./México/Canadá).
   6. Resolución de eliminatoria en TRES FASES: 90' -> prórroga (120') -> penales.
   7. Bracket determinista + simulación de Monte Carlo para probabilidad de título.

 Requisitos:  numpy, scipy
==============================================================================
"""
import numpy as np
from scipy.stats import poisson
from collections import Counter

# =============================================================================
# 1. RATINGS ELO PRE-TORNEO (fuente: worldfootball.elo)
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

NOMBRE = {
    "ESP":"España","ARG":"Argentina","FRA":"Francia","ENG":"Inglaterra","COL":"Colombia",
    "BRA":"Brasil","POR":"Portugal","NED":"P. Bajos","CRO":"Croacia","ECU":"Ecuador",
    "NOR":"Noruega","GER":"Alemania","SUI":"Suiza","URU":"Uruguay","TUR":"Turquía",
    "JPN":"Japón","SEN":"Senegal","BEL":"Bélgica","MEX":"México","PAR":"Paraguay",
    "AUT":"Austria","MAR":"Marruecos","CAN":"Canadá","AUS":"Australia","USA":"EE.UU.",
    "DZA":"Argelia","EGY":"Egipto","CPV":"Cabo Verde","GHA":"Ghana","BIH":"Bosnia",
    "COD":"R.D. Congo","CIV":"C. de Marfil","SWE":"Suecia","JPN":"Japón",
}

# =============================================================================
# 2. PARÁMETROS DEL MODELO
# =============================================================================
BASE_LAMBDA = 1.3942     # tasa base de goles por equipo
RHO         = -0.13      # parámetro de dependencia de Dixon-Coles (marcadores bajos)
K_ELO       = 40         # factor de aprendizaje del Elo dinámico
HOST        = {"USA", "MEX", "CAN"}   # anfitriones 2026
HOST_BONUS  = 35         # Elo extra por localía (bajado de 50 tras ver Canadá 0-3)
ET_FACTOR   = 1/3        # los 30' de prórroga => 1/3 del ritmo de goles de los 90'
PEN_DAMP    = 2000       # amortiguación del Elo en penales (tanda ~ moneda al aire)
PEN_CONV_F  = 0.77       # tasa de conversión del favorito en penales
PEN_CONV_U  = 0.74       # tasa de conversión del no-favorito

# =============================================================================
# 3. PARTIDOS JUGADOS (local, goles_local, goles_visita, visita)
#    Se usan para construir el Elo dinámico, en orden cronológico.
# =============================================================================
GRUPOS = [
    ("MEX",2,0,"RSA"),("KOR",2,1,"CZE"),("CZE",1,1,"RSA"),("MEX",1,0,"KOR"),
    ("CZE",0,3,"MEX"),("RSA",1,0,"KOR"),("CAN",1,1,"BIH"),("QAT",1,1,"SUI"),
    ("SUI",4,1,"BIH"),("CAN",6,0,"QAT"),("SUI",2,1,"CAN"),("BIH",3,1,"QAT"),
    ("BRA",1,1,"MAR"),("HAI",0,1,"SCO"),("SCO",0,1,"MAR"),("BRA",3,0,"HAI"),
    ("SCO",0,3,"BRA"),("MAR",4,2,"HAI"),("USA",4,1,"PAR"),("AUS",2,0,"TUR"),
    ("USA",2,0,"AUS"),("TUR",0,1,"PAR"),("TUR",3,2,"USA"),("PAR",0,0,"AUS"),
    ("GER",7,1,"CUW"),("CIV",1,0,"ECU"),("GER",2,1,"CIV"),("ECU",0,0,"CUW"),
    ("CUW",0,2,"CIV"),("ECU",2,1,"GER"),("NED",2,2,"JPN"),("SWE",5,1,"TUN"),
    ("NED",5,1,"SWE"),("TUN",0,4,"JPN"),("TUN",1,3,"NED"),("JPN",1,1,"SWE"),
    ("BEL",1,1,"EGY"),("IRN",2,2,"NZL"),("BEL",0,0,"IRN"),("NZL",1,3,"EGY"),
    ("EGY",1,1,"IRN"),("NZL",1,5,"BEL"),("ESP",0,0,"CPV"),("KSA",1,1,"URU"),
    ("ESP",4,0,"KSA"),("URU",2,2,"CPV"),("CPV",0,0,"KSA"),("URU",0,1,"ESP"),
    ("FRA",3,1,"SEN"),("IRQ",1,4,"NOR"),("FRA",3,0,"IRQ"),("NOR",3,2,"SEN"),
    ("NOR",1,4,"FRA"),("SEN",5,0,"IRQ"),("ARG",3,0,"DZA"),("AUT",3,1,"JOR"),
    ("ARG",2,0,"AUT"),("JOR",1,2,"DZA"),("DZA",3,3,"AUT"),("JOR",1,3,"ARG"),
    ("POR",1,1,"COD"),("UZB",1,3,"COL"),("POR",5,0,"UZB"),("COL",1,0,"COD"),
    ("COL",0,0,"POR"),("COD",3,1,"UZB"),("ENG",4,2,"CRO"),("GHA",1,0,"PAN"),
    ("ENG",0,0,"GHA"),("CRO",1,0,"PAN"),("PAN",0,2,"ENG"),("CRO",2,1,"GHA"),
]
# 16avos (dieciseisavos) — resultados reales (los empates se definieron por penales)
DIECISEISAVOS = [
    ("RSA",0,1,"CAN"),("BRA",2,1,"JPN"),("GER",1,1,"PAR"),("NED",1,1,"MAR"),
    ("CIV",1,2,"NOR"),("FRA",3,0,"SWE"),("MEX",2,0,"ECU"),("ENG",2,1,"COD"),
    ("BEL",3,2,"SEN"),("USA",2,0,"BIH"),("ESP",3,0,"AUT"),("POR",2,1,"CRO"),
    ("SUI",2,0,"DZA"),("AUS",1,1,"EGY"),("ARG",3,2,"CPV"),("COL",1,0,"GHA"),
]
# Octavos — resultados reales jugados hasta hoy
OCTAVOS_JUGADOS = [
    ("CAN",0,3,"MAR"),("PAR",0,1,"FRA"),("BRA",1,2,"NOR"),("MEX",2,3,"ENG"),
]
PLAYED = GRUPOS + DIECISEISAVOS + OCTAVOS_JUGADOS

# =============================================================================
# 4. ELO DINÁMICO
# =============================================================================
def build_dynamic_elo(matches, base=ELO0):
    """Actualiza el Elo partido a partido. K se amplifica en goleadas (|dif|>2)."""
    elo = dict(base)
    for h, hg, ag, a in matches:
        exp_h = 1.0 / (1.0 + 10 ** ((elo[a] - elo[h]) / 400))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = K_ELO * (1 + 0.1 * max(abs(hg - ag) - 2, 0))
        delta = k * (s_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
    return elo

# =============================================================================
# 5. DIXON-COLES + MEDIAS DE GOLES
# =============================================================================
def tau(x, y, la, mu, rho):
    """Factor de corrección de Dixon-Coles para marcadores bajos."""
    if x == 0 and y == 0: return 1 - la * mu * rho
    if x == 0 and y == 1: return 1 + la * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0

def dc_matrix(la, mu, rho=RHO, maxg=9):
    """Matriz de probabilidad de marcadores con corrección Dixon-Coles."""
    px = poisson.pmf(np.arange(maxg + 1), la)
    py = poisson.pmf(np.arange(maxg + 1), mu)
    M = np.outer(px, py)
    for x in range(2):
        for y in range(2):
            M[x, y] *= tau(x, y, la, mu, rho)
    return M / M.sum()

def lambdas(elo, a, b):
    """Medias de goles a partir de la diferencia de Elo (con bono de anfitrión)."""
    ea = elo[a] + (HOST_BONUS if a in HOST else 0)
    eb = elo[b] + (HOST_BONUS if b in HOST else 0)
    pw = 1.0 / (1.0 + 10 ** ((eb - ea) / 400))   # prob. de ganar del equipo A
    la = BASE_LAMBDA * (0.5 + pw)
    mu = BASE_LAMBDA * (1.5 - pw)
    return la, mu, ea, eb

# =============================================================================
# 6. RESOLUCIÓN EN TRES FASES (90' -> prórroga -> penales)
# =============================================================================
def penalty_shootout(rng, pc_fav=PEN_CONV_F, pc_und=PEN_CONV_U):
    """Simula una tanda: 5 tiros + muerte súbita. Devuelve (goles_fav, goles_und)."""
    a = b = 0
    for _ in range(5):
        a += rng.random() < pc_fav
        b += rng.random() < pc_und
    while a == b:
        a += rng.random() < pc_fav
        b += rng.random() < pc_und
    return a, b

def resolve_match(elo, a, b, rng=None):
    """
    Devuelve un dict con:
      fav          : código del favorito (quien más probablemente avanza)
      p_adv        : probabilidad de que el favorito avance (90'+prórroga+penales)
      sc90         : marcador 90' más probable con ganador
      p_prorroga   : prob. de llegar a la prórroga (empate a los 90')
      p_penales    : prob. de llegar a la tanda de penales
      pen_score    : marcador modal de penales (si rng se provee)
    """
    la, mu, ea, eb = lambdas(elo, a, b)
    M  = dc_matrix(la, mu)
    p1, pX, p2 = np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()
    # Prórroga: 30' a 1/3 del ritmo
    Me = dc_matrix(la * ET_FACTOR, mu * ET_FACTOR)
    e1, eX, e2 = np.tril(Me, -1).sum(), np.trace(Me), np.triu(Me, 1).sum()
    # Penales: casi moneda al aire (Elo fuertemente amortiguado)
    ppa = 1.0 / (1.0 + 10 ** ((eb - ea) / PEN_DAMP))
    adv_a = p1 + pX * (e1 + eX * ppa)
    adv_b = p2 + pX * (e2 + eX * (1 - ppa))
    fav = a if adv_a >= adv_b else b
    # Marcador 90' ganador más probable (para el favorito)
    mask = M.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (fav == a and i <= j) or (fav == b and j <= i):
                mask[i, j] = 0
    sc = np.unravel_index(mask.argmax(), mask.shape)
    pen_score = None
    if rng is not None:
        sims = [penalty_shootout(rng) for _ in range(4000)]
        pen_score = Counter(sims).most_common(1)[0][0]
    return dict(fav=fav, p_adv=max(adv_a, adv_b), sc90=f"{sc[0]}-{sc[1]}",
                p_prorroga=pX, p_penales=pX * eX, pen_score=pen_score)

def winner(elo, a, b):
    """Ganador determinista (el favorito)."""
    return resolve_match(elo, a, b)["fav"]

def simulate_match(elo, a, b, rng):
    """Simula UN partido de eliminación con las tres fases. Devuelve el ganador."""
    la, mu, ea, eb = lambdas(elo, a, b)
    ga, gb = rng.poisson(la), rng.poisson(mu)
    if ga > gb: return a
    if gb > ga: return b
    ga2, gb2 = rng.poisson(la * ET_FACTOR), rng.poisson(mu * ET_FACTOR)   # prórroga
    if ga2 > gb2: return a
    if gb2 > ga2: return b
    return a if rng.random() < 1.0/(1.0+10**((eb-ea)/PEN_DAMP)) else b     # penales

# =============================================================================
# 7. ESTRUCTURA DEL CUADRO (árbol oficial 2026)
#    Números de partido FIFA. Los W## que ya se jugaron van fijos.
# =============================================================================
# Ganadores ya conocidos (16avos completos + octavos jugados)
KNOWN = {89:"FRA", 90:"MAR", 91:"NOR", 92:"ENG"}   # octavos ya definidos
# Octavos que faltan: (num_partido) -> (equipoA, equipoB)
OCTAVOS_PEND = {93:("POR","ESP"), 94:("USA","BEL"), 95:("ARG","EGY"), 96:("SUI","COL")}
CUARTOS = [(97,89,90), (98,93,94), (99,91,92), (100,95,96)]
SEMIS   = [(101,97,98), (102,99,100)]
# FINAL = W101 vs W102

def deterministic_bracket(elo):
    """Avanza al favorito en cada llave y devuelve el cuadro completo + campeón."""
    w = dict(KNOWN)
    for m, (a, b) in OCTAVOS_PEND.items():
        w[m] = winner(elo, a, b)
    for m, x, y in CUARTOS + SEMIS:
        w[m] = winner(elo, w[x], w[y])
    champ = winner(elo, w[101], w[102])
    return w, champ

def monte_carlo(elo, n=30000, seed=11):
    """Probabilidad de título por simulación de Monte Carlo."""
    rng = np.random.default_rng(seed)
    ch = {t: 0 for t in elo}
    for _ in range(n):
        w = dict(KNOWN)
        for m, (a, b) in OCTAVOS_PEND.items():
            w[m] = simulate_match(elo, a, b, rng)
        for m, x, y in CUARTOS + SEMIS:
            w[m] = simulate_match(elo, w[x], w[y], rng)
        ch[simulate_match(elo, w[101], w[102], rng)] += 1
    return {t: c / n for t, c in ch.items()}

# =============================================================================
# 8. EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    elo = build_dynamic_elo(PLAYED)
    rng = np.random.default_rng(7)

    print("=== TOP-8 ELO DINÁMICO ACTUAL ===")
    for t in sorted(elo, key=elo.get, reverse=True)[:8]:
        print(f"  {NOMBRE.get(t,t):<12} {elo[t]:.0f}")

    print("\n=== OCTAVOS PENDIENTES (90' / prórroga / penales) ===")
    for m, (a, b) in OCTAVOS_PEND.items():
        r = resolve_match(elo, a, b, rng)
        print(f"  {NOMBRE[a]} - {NOMBRE[b]:<11} | 90':{r['sc90']} | "
              f"pen:{r['p_penales']*100:.0f}% ({r['pen_score'][0]}-{r['pen_score'][1]}) | "
              f"avanza {NOMBRE[r['fav']]} {r['p_adv']*100:.0f}%")

    print("\n=== CUADRO DETERMINISTA ===")
    w, champ = deterministic_bracket(elo)
    for m, x, y in CUARTOS:
        print(f"  4tos: {NOMBRE[w[x]]} - {NOMBRE[w[y]]:<11} -> {NOMBRE[w[m]]}")
    for m, x, y in SEMIS:
        print(f"  Semi: {NOMBRE[w[x]]} - {NOMBRE[w[y]]:<11} -> {NOMBRE[w[m]]}")
    print(f"  FINAL: {NOMBRE[w[101]]} - {NOMBRE[w[102]]} -> CAMPEÓN {NOMBRE[champ]}")

    print("\n=== PROBABILIDAD DE TÍTULO (Monte Carlo 30k) ===")
    mc = monte_carlo(elo)
    for t, p in sorted(mc.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {NOMBRE.get(t,t):<12} {p*100:.1f}%")
