# Resultados Liga 1 Perú 2026 (fuentes: futbolperuano.com, depor.com, liga1.pe; consultados 07-nov-2026)
# Formato: (torneo, fecha, local, visitante, goles_local, goles_visitante)
# Apertura: fechas 1-14 completas + 3 partidos de la fecha 15 (el resto de la 15, 16 y 17 no se pudo obtener).
# Clausura: fechas 1-15 completas.
import re

def _parse(torneo, texto):
    out = []
    for ln in texto.strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"F(\d+):\s*(.+?)\s+(\d+)-(\d+)\s+(.+)$", ln)
        f, h, gh, ga, a = m.groups()
        out.append((torneo, int(f), h.strip(), a.strip(), int(gh), int(ga)))
    return out

APERTURA = """
F1: Sport Huancayo 1-2 Alianza Lima
F1: UTC 2-0 Atlético Grau
F1: Sport Boys 1-1 Los Chankas
F1: Juan Pablo II 3-3 FC Cajamarca
F1: Melgar 2-0 Cienciano
F1: Dep. Garcilaso 1-1 Sporting Cristal
F1: Alianza Atlético 1-0 Cusco FC
F1: Universitario 2-0 ADT
F1: Comerciantes Unidos 1-0 CD Moquegua
F2: CD Moquegua 2-3 UTC
F2: ADT 1-0 Sport Boys
F2: Atlético Grau 0-0 Sport Huancayo
F2: Cusco FC 1-1 Universitario
F2: Sporting Cristal 1-2 Melgar
F2: Los Chankas 1-0 Alianza Atlético
F2: Cienciano 6-1 Juan Pablo II
F2: Alianza Lima 2-1 Comerciantes Unidos
F2: FC Cajamarca 1-1 Dep. Garcilaso
F3: UTC 1-0 Cusco FC
F3: Universitario 2-1 Cienciano
F3: Dep. Garcilaso 1-0 ADT
F3: Juan Pablo II 0-2 Sporting Cristal
F3: Sport Boys 1-0 Atlético Grau
F3: Alianza Atlético 0-0 Alianza Lima
F3: Sport Huancayo 2-0 FC Cajamarca
F3: Comerciantes Unidos 1-1 Los Chankas
F3: Melgar 4-0 CD Moquegua
F4: Alianza Lima 1-0 Sport Boys
F4: FC Cajamarca 3-1 Melgar
F4: Sporting Cristal 2-2 Universitario
F4: Cienciano 1-1 Alianza Atlético
F4: ADT 2-2 UTC
F4: Los Chankas 3-2 Sport Huancayo
F4: Atlético Grau 1-2 Juan Pablo II
F4: Cusco FC 3-1 Comerciantes Unidos
F4: CD Moquegua 1-0 Dep. Garcilaso
F5: Alianza Atlético 0-0 ADT
F5: Melgar 1-2 Los Chankas
F5: Dep. Garcilaso 2-3 Cienciano
F5: Sport Huancayo 2-1 Sporting Cristal
F5: UTC 0-1 Alianza Lima
F5: Sport Boys 0-0 CD Moquegua
F5: Comerciantes Unidos 3-1 Atlético Grau
F5: Juan Pablo II 2-1 Cusco FC
F5: Universitario 1-0 FC Cajamarca
F6: Atlético Grau 1-0 FC Cajamarca
F6: CD Moquegua 2-1 Sport Huancayo
F6: Comerciantes Unidos 1-2 UTC
F6: Sporting Cristal 3-1 Alianza Atlético
F6: Cusco FC 1-0 Dep. Garcilaso
F6: ADT 2-3 Juan Pablo II
F6: Los Chankas 3-1 Universitario
F6: Cienciano 3-1 Sport Boys
F6: Alianza Lima 3-1 Melgar
F7: FC Cajamarca 3-4 Comerciantes Unidos
F7: Sport Huancayo 0-1 ADT
F7: Juan Pablo II 2-3 Los Chankas
F7: Dep. Garcilaso 1-1 Alianza Lima
F7: Universitario 2-0 UTC
F7: Sporting Cristal 3-0 Sport Boys
F7: Melgar 0-0 Atlético Grau
F7: Alianza Atlético 3-0 CD Moquegua
F7: Cusco FC 1-2 Cienciano
F8: ADT 1-1 Melgar
F8: Cienciano 3-0 FC Cajamarca
F8: CD Moquegua 1-2 Cusco FC
F8: Comerciantes Unidos 0-0 Universitario
F8: Los Chankas 3-2 Sporting Cristal
F8: Alianza Lima 2-0 Juan Pablo II
F8: UTC 1-1 Alianza Atlético
F8: Atlético Grau 0-0 Dep. Garcilaso
F8: Sport Boys 3-0 Sport Huancayo
F9: Sporting Cristal 1-2 CD Moquegua
F9: Alianza Atlético 2-2 Atlético Grau
F9: Cienciano 3-2 ADT
F9: Dep. Garcilaso 3-2 Sport Boys
F9: Melgar 1-3 Cusco FC
F9: Universitario 1-0 Alianza Lima
F9: Sport Huancayo 1-1 Comerciantes Unidos
F9: Juan Pablo II 2-1 UTC
F9: FC Cajamarca 0-1 Los Chankas
F10: CD Moquegua 3-0 Juan Pablo II
F10: ADT 0-1 Alianza Lima
F10: Sport Boys 1-3 Melgar
F10: UTC 2-2 Sport Huancayo
F10: Comerciantes Unidos 1-0 Alianza Atlético
F10: Atlético Grau 4-1 Sporting Cristal
F10: Cusco FC 1-0 FC Cajamarca
F10: Universitario 4-1 Dep. Garcilaso
F10: Los Chankas 1-0 Cienciano
F11: Los Chankas 2-0 Atlético Grau
F11: FC Cajamarca 2-2 ADT
F11: Juan Pablo II 2-2 Comerciantes Unidos
F11: Sport Huancayo 2-1 Dep. Garcilaso
F11: Alianza Lima 8-0 Cusco FC
F11: Sporting Cristal 3-2 UTC
F11: Alianza Atlético 0-1 Sport Boys
F11: Melgar 2-1 Universitario
F11: Cienciano 2-1 CD Moquegua
F12: CD Moquegua 2-1 FC Cajamarca
F12: Comerciantes Unidos 1-0 Sporting Cristal
F12: Sport Boys 1-1 Juan Pablo II
F12: UTC 2-2 Cienciano
F12: Cusco FC 2-1 Sport Huancayo
F12: Atlético Grau 0-1 Alianza Lima
F12: Universitario 1-2 Alianza Atlético
F12: ADT 4-0 Los Chankas
F12: Dep. Garcilaso 1-0 Melgar
F13: FC Cajamarca 3-2 Sport Boys
F13: ADT 1-0 Atlético Grau
F13: Cienciano 1-0 Comerciantes Unidos
F13: Alianza Lima 2-1 CD Moquegua
F13: Sporting Cristal 2-2 Cusco FC
F13: Juan Pablo II 1-4 Universitario
F13: Alianza Atlético 4-1 Sport Huancayo
F13: Melgar 2-0 UTC
F13: Los Chankas 0-1 Dep. Garcilaso
F14: UTC 1-2 FC Cajamarca
F14: CD Moquegua 2-2 ADT
F14: Atlético Grau 1-2 Cienciano
F14: Sport Huancayo 2-1 Juan Pablo II
F14: Comerciantes Unidos 1-1 Melgar
F14: Cusco FC 1-1 Los Chankas
F14: Dep. Garcilaso 2-1 Alianza Atlético
F14: Sport Boys 0-0 Universitario
F14: Alianza Lima 1-1 Sporting Cristal
F15: FC Cajamarca 3-1 Sporting Cristal
F15: Universitario 0-1 Atlético Grau
F15: Melgar 4-1 Sport Huancayo
"""

CLAUSURA = """
F1: FC Cajamarca 2-2 Juan Pablo II
F1: Sporting Cristal 1-0 Dep. Garcilaso
F1: Cienciano 1-3 Melgar
F1: CD Moquegua 2-2 Comerciantes Unidos
F1: ADT 1-2 Universitario
F1: Atlético Grau 3-1 UTC
F1: Cusco FC 4-1 Alianza Atlético
F1: Los Chankas 1-0 Sport Boys
F1: Alianza Lima 2-1 Sport Huancayo
F2: Alianza Atlético 3-1 Los Chankas
F2: Universitario 2-1 Cusco FC
F2: Sport Boys 2-0 ADT
F2: Sport Huancayo 2-0 Atlético Grau
F2: Juan Pablo II 2-0 Cienciano
F2: Melgar 2-1 Sporting Cristal
F2: UTC 0-0 CD Moquegua
F2: Comerciantes Unidos 2-2 Alianza Lima
F2: Dep. Garcilaso 2-1 FC Cajamarca
F3: Los Chankas 3-2 Comerciantes Unidos
F3: FC Cajamarca 0-2 Sport Huancayo
F3: Atlético Grau 0-1 Sport Boys
F3: Cusco FC 4-1 UTC
F3: Alianza Lima 3-1 Alianza Atlético
F3: Sporting Cristal 2-0 Juan Pablo II
F3: ADT 0-1 Dep. Garcilaso
F3: CD Moquegua 0-0 Melgar
F3: Cienciano 2-0 Universitario
F4: Alianza Atlético 1-0 Cienciano
F4: Sport Huancayo 1-1 Los Chankas
F4: Comerciantes Unidos 3-1 Cusco FC
F4: Universitario 1-1 Sporting Cristal
F4: UTC 1-1 ADT
F4: Juan Pablo II 0-0 Atlético Grau
F4: Sport Boys 1-1 Alianza Lima
F4: Dep. Garcilaso 2-0 CD Moquegua
F4: Melgar 2-4 FC Cajamarca
F5: Atlético Grau 1-0 Comerciantes Unidos
F5: Cusco FC 2-0 Juan Pablo II
F5: ADT 1-1 Alianza Atlético
F5: Los Chankas 0-3 Melgar
F5: Alianza Lima 1-0 UTC
F5: Sporting Cristal 4-1 Sport Huancayo
F5: CD Moquegua 1-1 Sport Boys
F5: Cienciano 0-0 Dep. Garcilaso
F5: FC Cajamarca 2-3 Universitario
F6: FC Cajamarca 1-3 Atlético Grau
F6: Alianza Atlético 3-1 Sporting Cristal
F6: Sport Huancayo 0-2 CD Moquegua
F6: UTC 1-3 Comerciantes Unidos
F6: Juan Pablo II 5-1 ADT
F6: Dep. Garcilaso 4-1 Cusco FC
F6: Sport Boys 5-1 Cienciano
F6: Melgar 1-1 Alianza Lima
F6: Universitario 3-0 Los Chankas
F7: Comerciantes Unidos 2-3 FC Cajamarca
F7: Los Chankas 3-4 Juan Pablo II
F7: UTC 2-4 Universitario
F7: Alianza Lima 0-1 Dep. Garcilaso
F7: CD Moquegua 1-4 Alianza Atlético
F7: ADT 1-3 Sport Huancayo
F7: Sport Boys 1-0 Sporting Cristal
F7: Cienciano 1-2 Cusco FC
F7: Atlético Grau 1-0 Melgar
F8: FC Cajamarca 2-5 Cienciano
F8: Alianza Atlético 3-2 UTC
F8: Sport Huancayo 2-1 Sport Boys
F8: Cusco FC 2-0 CD Moquegua
F8: Sporting Cristal 5-0 Los Chankas
F8: Dep. Garcilaso 4-2 Atlético Grau
F8: Juan Pablo II 1-1 Alianza Lima
F8: Universitario 2-2 Comerciantes Unidos
F8: Melgar 3-0 ADT
F9: UTC 1-2 Juan Pablo II
F9: Cusco FC 0-1 Melgar
F9: Los Chankas 1-2 FC Cajamarca
F9: Comerciantes Unidos 0-1 Sport Huancayo
F9: Alianza Lima 1-2 Universitario
F9: CD Moquegua 1-2 Sporting Cristal
F9: Sport Boys 1-0 Dep. Garcilaso
F9: Atlético Grau 3-1 Alianza Atlético
F9: ADT 2-1 Cienciano
F10: Alianza Atlético 2-1 Comerciantes Unidos
F10: Alianza Lima 2-0 ADT
F10: FC Cajamarca 1-2 Cusco FC
F10: Sporting Cristal 3-1 Atlético Grau
F10: Melgar 1-1 Sport Boys
F10: Dep. Garcilaso 4-0 Universitario
F10: Sport Huancayo 3-2 UTC
F10: Juan Pablo II 4-0 CD Moquegua
F10: Cienciano 1-1 Los Chankas
F11: CD Moquegua 0-2 Cienciano
F11: Atlético Grau 0-0 Los Chankas
F11: Dep. Garcilaso 0-1 Sport Huancayo
F11: ADT 0-0 FC Cajamarca
F11: Comerciantes Unidos 1-1 Juan Pablo II
F11: Universitario 2-2 Melgar
F11: UTC 0-1 Sporting Cristal
F11: Sport Boys 1-3 Alianza Atlético
F11: Cusco FC 1-2 Alianza Lima
F12: Sport Huancayo 2-1 Cusco FC
F12: Los Chankas 2-1 ADT
F12: Juan Pablo II 2-0 Sport Boys
F12: Melgar 4-2 Dep. Garcilaso
F12: Alianza Lima 3-1 Atlético Grau
F12: Sporting Cristal 4-2 Comerciantes Unidos
F12: FC Cajamarca 0-1 CD Moquegua
F12: Alianza Atlético 1-4 Universitario
F12: Cienciano 1-0 UTC
F13: UTC 1-2 Melgar
F13: Atlético Grau 1-1 ADT
F13: Sport Huancayo 4-1 Alianza Atlético
F13: Cusco FC 3-1 Sporting Cristal
F13: Universitario 3-0 Juan Pablo II
F13: Comerciantes Unidos 3-1 Cienciano
F13: CD Moquegua 0-1 Alianza Lima
F13: Sport Boys 4-2 FC Cajamarca
F13: Dep. Garcilaso 4-1 Los Chankas
F14: Juan Pablo II 2-0 Sport Huancayo
F14: ADT 2-0 CD Moquegua
F14: FC Cajamarca 1-3 UTC
F14: Alianza Atlético 3-2 Dep. Garcilaso
F14: Cienciano 3-0 Atlético Grau
F14: Los Chankas 0-1 Cusco FC
F14: Sporting Cristal 3-2 Alianza Lima
F14: Melgar 1-1 Comerciantes Unidos
F14: Universitario 2-2 Sport Boys
F15: Alianza Atlético 1-0 Juan Pablo II
F15: CD Moquegua 0-0 Los Chankas
F15: Cusco FC 0-1 Sport Boys
F15: Atlético Grau 1-3 Universitario
F15: Sport Huancayo 2-0 Melgar
F15: UTC 2-3 Dep. Garcilaso
F15: Sporting Cristal 5-0 FC Cajamarca
F15: Comerciantes Unidos 2-2 ADT
F15: Alianza Lima 1-0 Cienciano
F16: Atlético Grau 3-0 Cusco FC
F16: ADT 1-1 Sporting Cristal
F16: FC Cajamarca 2-0 Alianza Atlético
F16: Melgar 2-1 Juan Pablo II
F16: Cienciano 1-2 Sport Huancayo
F16: Los Chankas 1-1 Alianza Lima
F16: Sport Boys 0-0 UTC
F16: Dep. Garcilaso 0-0 Comerciantes Unidos
F16: Universitario 2-2 CD Moquegua
"""

PARTIDOS = _parse("A", APERTURA) + _parse("C", CLAUSURA)
