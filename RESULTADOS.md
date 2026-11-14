# Resultados del backtest walk-forward

Generado por `python backtest.py`. Cada ronda se predice usando solo rondas anteriores (entrenamiento, hiperparámetros y peso del ensamble).

- **Acierto 1X2**: acertó local / empate / visita. **Marcador exacto**: el marcador más probable fue el real.
- **LogLoss / RPS**: calidad de las probabilidades; **menor es mejor**. "LogLoss marcador" mide la probabilidad dada al marcador real.

## Liga 1 Perú 2026

Rondas evaluadas: A6 a C16.

| Modelo | Partidos | Acierto 1X2 | Marcador exacto | Marcador en top-3 | LogLoss 1X2 | RPS | LogLoss marcador |
|---|---|---|---|---|---|---|---|
| Base (promedio de goles) | 228 | 50.9% | 28/228 (12.3%) | 27.2% | 1.038 | 0.225 | 2.998 |
| Modelo original del repositorio | 228 | 53.1% | 19/228 (8.3%) | 27.6% | 1.021 | 0.217 | 2.990 |
| Estadístico mejorado (HP fijos) | 228 | 52.2% | 21/228 (9.2%) | 28.9% | 1.016 | 0.217 | 2.981 |
| Estadístico mejorado (HP auto) | 228 | 53.1% | 22/228 (9.6%) | 29.8% | 1.017 | 0.217 | 2.992 |
| Deep: red preentrenada, sin efectos de equipo | 228 | 54.4% | 25/228 (11.0%) | 26.3% | 1.016 | 0.216 | 2.994 |
| Deep learning (afinado, auto) | 228 | 53.9% | 26/228 (11.4%) | 26.8% | 1.017 | 0.217 | 3.003 |
| ENSAMBLE estadístico + deep | 228 | 52.6% | 25/228 (11.0%) | 29.4% | 1.019 | 0.217 | 2.998 |
| Ensamble 50/50 fijo | 228 | 53.1% | 27/228 (11.8%) | 28.1% | 1.015 | 0.216 | 2.993 |

### Misma ventana que el informe original (Clausura 11 a 15, 45 partidos)

| Modelo | Partidos | Acierto 1X2 | Marcador exacto | Marcador en top-3 | LogLoss 1X2 | RPS | LogLoss marcador |
|---|---|---|---|---|---|---|---|
| Base (promedio de goles) | 45 | 48.9% | 3/45 (6.7%) | 17.8% | 1.051 | 0.236 | 3.061 |
| Modelo original del repositorio | 45 | 60.0% | 2/45 (4.4%) | 20.0% | 0.980 | 0.211 | 2.992 |
| Estadístico mejorado (HP fijos) | 45 | 57.8% | 2/45 (4.4%) | 20.0% | 0.975 | 0.209 | 2.991 |
| Estadístico mejorado (HP auto) | 45 | 57.8% | 2/45 (4.4%) | 20.0% | 0.969 | 0.207 | 2.989 |
| Deep: red preentrenada, sin efectos de equipo | 45 | 60.0% | 2/45 (4.4%) | 24.4% | 0.965 | 0.205 | 2.992 |
| Deep learning (afinado, auto) | 45 | 60.0% | 2/45 (4.4%) | 24.4% | 0.965 | 0.205 | 2.992 |
| ENSAMBLE estadístico + deep | 45 | 57.8% | 3/45 (6.7%) | 22.2% | 0.974 | 0.208 | 3.000 |
| Ensamble 50/50 fijo | 45 | 60.0% | 2/45 (4.4%) | 22.2% | 0.965 | 0.205 | 2.985 |

### Comparación pareada (mismos partidos)

- **Estadístico mejorado (HP auto)** vs original: LogLoss marcador +0.002 (IC 95 %: -0.013 a +0.019); en 1X2 acierta 1 que el original falla y falla 1 que el original acierta.
- **Deep learning (afinado, auto)** vs original: LogLoss marcador +0.013 (IC 95 %: -0.011 a +0.038); en 1X2 acierta 6 que el original falla y falla 4 que el original acierta.
- **ENSAMBLE estadístico + deep** vs original: LogLoss marcador +0.008 (IC 95 %: -0.008 a +0.025); en 1X2 acierta 1 que el original falla y falla 2 que el original acierta.

Elección para la próxima ronda (la usa `pronosticar.py`): hiperparámetros elegidos {'xi': 0.03, 'ridge': 16.0, 'ridge_h': 8.0}, α=0.75; variante deep elegida = cero; peso del modelo estadístico en el ensamble = 0.75.

## Mundial 2026

Rondas evaluadas: J1 a QF.

| Modelo | Partidos | Acierto 1X2 | Marcador exacto | Marcador en top-3 | LogLoss 1X2 | RPS | LogLoss marcador |
|---|---|---|---|---|---|---|---|
| Base (promedio de goles) | 97 | 49.5% | 10/97 (10.3%) | 24.7% | 1.090 | 0.237 | 3.246 |
| Modelo original del repositorio | 97 | 62.9% | 9/97 (9.3%) | 32.0% | 0.888 | 0.168 | 2.925 |
| Estadístico mejorado (HP fijos) | 97 | 60.8% | 12/97 (12.4%) | 40.2% | 0.872 | 0.164 | 2.924 |
| Estadístico mejorado (HP auto) | 97 | 60.8% | 11/97 (11.3%) | 37.1% | 0.874 | 0.165 | 2.927 |
| Deep: red preentrenada, sin efectos de equipo | 97 | 63.9% | 14/97 (14.4%) | 35.1% | 0.880 | 0.160 | 2.866 |
| Deep learning (afinado, auto) | 97 | 63.9% | 13/97 (13.4%) | 34.0% | 0.877 | 0.159 | 2.878 |
| ENSAMBLE estadístico + deep | 97 | 63.9% | 17/97 (17.5%) | 34.0% | 0.859 | 0.158 | 2.880 |
| Ensamble 50/50 fijo | 97 | 62.9% | 17/97 (17.5%) | 36.1% | 0.864 | 0.160 | 2.884 |

### Comparación pareada (mismos partidos)

- **Estadístico mejorado (HP auto)** vs original: LogLoss marcador +0.002 (IC 95 %: -0.063 a +0.068); en 1X2 acierta 3 que el original falla y falla 5 que el original acierta.
- **Deep learning (afinado, auto)** vs original: LogLoss marcador -0.047 (IC 95 %: -0.159 a +0.067); en 1X2 acierta 5 que el original falla y falla 4 que el original acierta.
- **ENSAMBLE estadístico + deep** vs original: LogLoss marcador -0.045 (IC 95 %: -0.131 a +0.044); en 1X2 acierta 5 que el original falla y falla 4 que el original acierta.

Elección para la próxima ronda (la usa `pronosticar.py`): hiperparámetros elegidos {'xi': 0.03, 'ridge': 4.0, 'ridge_h': 10000.0}, α=0.75; variante deep elegida = cero; peso del modelo estadístico en el ensamble = 0.25.

