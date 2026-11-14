"""
Resultados del Mundial 2026 cargados en el repositorio (extraídos de auditoria_walkforward.py).
Formato de cada partido: (equipo_1, goles_1, goles_2, equipo_2). Marcador tal como estaba cargado.
ELO0 = rating Elo previo al torneo. PEN = quién avanzó en los empates de eliminatoria.
"""
ELO0={"ESP":2171,"ARG":2113,"FRA":2063,"ENG":2042,"COL":1998,"BRA":1979,"POR":1976,"NED":1959,"CRO":1933,"ECU":1933,"NOR":1922,"GER":1910,"SUI":1897,"URU":1890,"TUR":1880,"JPN":1879,"SEN":1869,"BEL":1849,"MEX":1834,"PAR":1833,"AUT":1818,"MAR":1806,"CAN":1806,"SCO":1790,"KOR":1784,"AUS":1774,"IRN":1754,"USA":1747,"PAN":1743,"UZB":1735,"CZE":1731,"DZA":1728,"JOR":1691,"SWE":1660,"EGY":1660,"COD":1639,"CIV":1637,"TUN":1614,"KSA":1592,"NZL":1586,"IRQ":1583,"BIH":1571,"CPV":1561,"HAI":1542,"RSA":1529,"GHA":1509,"CUW":1467,"QAT":1427}
ANFITRIONES = ("USA", "MEX", "CAN")
J1 = [("MEX",2,0,"RSA"),("KOR",2,1,"CZE"),("CAN",1,1,"BIH"),("QAT",1,1,"SUI"),("BRA",1,1,"MAR"),("HAI",0,1,"SCO"),("USA",4,1,"PAR"),("AUS",2,0,"TUR"),("GER",7,1,"CUW"),("CIV",1,0,"ECU"),("NED",2,2,"JPN"),("SWE",5,1,"TUN"),("BEL",1,1,"EGY"),("IRN",2,2,"NZL"),("ESP",0,0,"CPV"),("KSA",1,1,"URU"),("FRA",3,1,"SEN"),("IRQ",1,4,"NOR"),("ARG",3,0,"DZA"),("AUT",3,1,"JOR"),("POR",1,1,"COD"),("UZB",1,3,"COL"),("ENG",4,2,"CRO"),("GHA",1,0,"PAN")]
J2 = [("CZE",1,1,"RSA"),("MEX",1,0,"KOR"),("SUI",4,1,"BIH"),("CAN",6,0,"QAT"),("SCO",0,1,"MAR"),("BRA",3,0,"HAI"),("USA",2,0,"AUS"),("TUR",0,1,"PAR"),("GER",2,1,"CIV"),("ECU",0,0,"CUW"),("NED",5,1,"SWE"),("TUN",0,4,"JPN"),("BEL",0,0,"IRN"),("NZL",1,3,"EGY"),("ESP",4,0,"KSA"),("URU",2,2,"CPV"),("FRA",3,0,"IRQ"),("NOR",3,2,"SEN"),("ARG",2,0,"AUT"),("JOR",1,2,"DZA"),("POR",5,0,"UZB"),("COL",1,0,"COD"),("ENG",0,0,"GHA"),("CRO",1,0,"PAN")]
J3 = [("CZE",0,3,"MEX"),("RSA",1,0,"KOR"),("SUI",2,1,"CAN"),("BIH",3,1,"QAT"),("SCO",0,3,"BRA"),("MAR",4,2,"HAI"),("TUR",3,2,"USA"),("PAR",0,0,"AUS"),("CUW",0,2,"CIV"),("ECU",2,1,"GER"),("TUN",1,3,"NED"),("JPN",1,1,"SWE"),("EGY",1,1,"IRN"),("NZL",1,5,"BEL"),("CPV",0,0,"KSA"),("URU",0,1,"ESP"),("NOR",1,4,"FRA"),("SEN",5,0,"IRQ"),("DZA",3,3,"AUT"),("JOR",1,3,"ARG"),("COL",0,0,"POR"),("COD",3,1,"UZB"),("PAN",0,2,"ENG"),("CRO",2,1,"GHA")]
R32 = [("RSA",0,1,"CAN"),("BRA",2,1,"JPN"),("GER",1,1,"PAR"),("NED",1,1,"MAR"),("CIV",1,2,"NOR"),("FRA",3,0,"SWE"),("MEX",2,0,"ECU"),("ENG",2,1,"COD"),("BEL",3,2,"SEN"),("USA",2,0,"BIH"),("ESP",3,0,"AUT"),("POR",2,1,"CRO"),("SUI",2,0,"DZA"),("AUS",1,1,"EGY"),("ARG",3,2,"CPV"),("COL",1,0,"GHA")]
R16 = [("CAN",0,3,"MAR"),("PAR",0,1,"FRA"),("BRA",1,2,"NOR"),("MEX",2,3,"ENG"),("POR",0,1,"ESP"),("USA",1,4,"BEL"),("ARG",3,2,"EGY"),("SUI",0,0,"COL")]
QF = [("FRA",2,0,"MAR")]
PEN={("GER","PAR"):"PAR",("NED","MAR"):"MAR",("AUS","EGY"):"EGY",("SUI","COL"):"SUI"}
RONDAS = [("J1", J1), ("J2", J2), ("J3", J3), ("R32", R32), ("R16", R16), ("QF", QF)]
