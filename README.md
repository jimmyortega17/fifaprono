# fifaprono

Pronósticos de fútbol (Liga 1 Perú 2026 y Mundial 2026): probabilidades 1X2, marcador exacto y mercados.

## Modelos nuevos (noviembre 2026)

| Archivo | Qué hace |
|---|---|
| `nucleo.py` | Carga los resultados del repositorio, matriz de marcadores Poisson + Dixon-Coles, Elo y métricas. |
| `modelo_mejorado.py` | Modelo estadístico mejorado: Elo dinámico + ataque/defensa con contracción, decaimiento temporal, ancla en Elo previo y ventaja de casa propia de cada equipo. Sirve igual para Liga 1 y Mundial. |
| `modelo_deep.py` | Red neuronal (PyTorch) que predice la distribución completa del marcador. Núcleo preentrenado + adaptador por liga. |
| `preentrenar.py` | Preentrena la red con ~46 000 partidos internacionales (1950 a 10-jun-2026, datos públicos CC0). Genera `pesos/preentrenado.pt` (ya incluido). |
| `backtest.py` | Backtest walk-forward de todos los modelos, incluido el original. Escribe `RESULTADOS.md`, `resultados_backtest.json` y `seleccion.json`. |
| `pronosticar.py` | Pronostica partidos no jugados con el ensamble (por defecto, fecha 17 del Clausura). |
| `datos_mundial2026.py` | Resultados del Mundial que estaban dentro de `auditoria_walkforward.py`, como módulo de datos. |

```bash
pip install -r requirements.txt
```

```bash
python backtest.py
```

```bash
python pronosticar.py
```

`python preentrenar.py` solo hace falta si quieres regenerar los pesos (descarga el histórico con git).

## Qué dice el backtest

Tablas completas en [`RESULTADOS.md`](RESULTADOS.md). En resumen:

- **Mundial (97 partidos):** el ensamble acierta 17 marcadores exactos contra 9 del modelo original y mejora
  las probabilidades (LogLoss 1X2 0.859 contra 0.888). El acierto 1X2 es casi el mismo (63.9 % contra 62.9 %).
- **Liga 1 (228 partidos):** todos los modelos quedan prácticamente empatados (52 a 54 % en 1X2). El ensamble
  acierta 25 marcadores exactos contra 19 del original, pero ninguno supera con claridad a pronosticar siempre
  el marcador promedio (28).
- **Ninguna diferencia es estadísticamente concluyente** con tan pocos partidos: los intervalos de confianza
  de la comparación pareada incluyen el cero.
- Una red entrenada **solo** con los partidos del repositorio se sobreajusta y pronostica mucho peor
  (46 % en Liga 1, 43 % en el Mundial). Por eso se preentrena con datos externos.

El marcador exacto tiene un techo bajo: el marcador más probable de un partido rara vez pasa del 12 al 14 %.

## Reglas del backtest (sin trampa)

Para predecir una ronda solo se usan rondas anteriores: entrenamiento, hiperparámetros y peso del ensamble.
El preentrenamiento usa únicamente partidos anteriores al inicio del Mundial, y ninguno de Liga 1.

## Scripts anteriores

Los scripts originales (`modelo_liga1_walkforward.py`, `modelo_mundial2026_v2.py`, `pronostico*.py`, etc.) se
conservan sin cambios. `backtest.py` usa `modelo_liga1_walkforward.py` como referencia.
