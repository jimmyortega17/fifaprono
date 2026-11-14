"""
Factor de motivación (quién juega por algo) para la fecha 17 del Clausura 2026 de la Liga 1.

Fuentes de la situación (tablas tras la fecha 16): Acumulada (33 PJ) y Clausura.
Reglas (Libero, 05-nov-2026): descienden 17.º y 18.º del Acumulada; Sudamericana = 5.º a 8.º del Acumulada;
Libertadores/título por playoffs (Alianza Lima ya clasificado). Título Clausura: Huancayo 34, Cristal 32, Universitario 32.

Escala de motivación m: 1.0 = se juega algo importante, 0.5 = algo menor, 0.0 = nada en juego.
Efecto en el modelo: lambda_equipo *= exp(BETA * (m_equipo - m_rival)).
BETA es un SUPUESTO prudente (no se estimó con datos de Liga 1: la muestra es muy pequeña).
"""
import numpy as np

BETA = 0.10   # con diferencia máxima de motivación (1 vs 0): ±10 % en los goles esperados

# --- fecha 17 (Clausura): motivación por equipo y por qué ---
STAKES = {
    'Universitario':       (1.0, 'Clausura: 32 pts, a 2 de Huancayo (juegan entre sí); 2.º del Acumulado asegurado'),
    'Sport Huancayo':      (1.0, 'Clausura: líder con 34 pts; además 8.º del Acumulado (50) con Alianza Atlético a 1 pt'),
    'Sporting Cristal':    (1.0, 'Clausura: 32 pts, a 2 del líder; 5.º del Acumulado (51)'),
    'Cienciano':           (1.0, 'Sudamericana: 7.º del Acumulado (50), perseguido por Alianza Atlético (49) y Chankas (48)'),
    'Cusco FC':            (1.0, 'Sudamericana: 6.º del Acumulado (51), pelea cupo 5.º a 8.º'),
    'Alianza Atlético':    (1.0, 'Sudamericana: 9.º del Acumulado (49), a 1 pt del 8.º'),
    'Los Chankas':         (1.0, 'Sudamericana: 10.º del Acumulado (48), a 2 pts del 8.º'),
    'ADT':                 (1.0, 'Descenso: 15.º del Acumulado (31), a 1 pt del 16.º'),
    'CD Moquegua':         (1.0, 'Descenso: 16.º del Acumulado (30)'),
    'FC Cajamarca':        (1.0, 'Descenso: 17.º del Acumulado (29), zona de descenso'),
    'Melgar':              (0.5, 'Libertadores: 3.º del Acumulado (57), solo disputa 3.º vs 4.º con Garcilaso (55)'),
    'Dep. Garcilaso':      (0.5, 'Libertadores: 4.º del Acumulado (55); 4.º asegurado, podría subir a 3.º'),
    'Alianza Lima':        (0.0, '1.º del Acumulado asegurado (69) y fuera de la pelea del Clausura (29); posible rotación antes de los playoffs'),
    'Juan Pablo II':       (0.0, '12.º del Acumulado (41): sin descenso ni cupo posibles'),
    'Sport Boys':          (0.0, '11.º del Acumulado (42): sin descenso ni cupo posibles'),
    'Comerciantes Unidos': (0.0, '13.º (37): descenso descartado (ADT 31 + 3 = 34 máx.)'),
    'Atlético Grau':       (0.0, '14.º (37): descenso descartado'),
    'UTC':                 (0.0, 'Ya descendido (14 pts, máx. 17)'),
}


def m(team):
    return STAKES[team][0]


def adjust(lh, la, home, away, beta=BETA):
    d = m(home) - m(away)
    return lh * np.exp(beta * d), la * np.exp(-beta * d)
