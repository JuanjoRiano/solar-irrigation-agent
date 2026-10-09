# Parcial integrador: código, datos y resultados

Minimax + STRIPS + red bayesiana. Sector: programación de riego con energía solar.

## Ejecutar

Requiere Python 3.10 o superior. Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 agent.py
python3 test_agent.py
```

En Windows, activar con `.venv\Scripts\activate`. La ejecución usa la copia de datos incluida y no requiere Internet. Para ejecutar solamente cálculos y log, sin instalar matplotlib: `python3 agent.py --no-plots`.

```bash
python3 agent.py --date 2025-01-24
python3 agent.py --date 2025-01-24 --prior 0.05 --output demo_prior_bajo
python3 agent.py --date 2025-01-24 --prior 0.60 --output demo_prior_alto
```

`--output` separa las salidas de cada demo. El posterior base usa las frecuencias ajustadas; `--prior` es un experimento hipotético. Fechas válidas: 2025-01-01 a 2025-12-30. `--repeats 31` controla repeticiones del benchmark; se informa la mediana en milisegundos. Los tiempos dependen del equipo y la carga.

## Qué es real y qué es una decisión de diseño

**Datos reales de una fuente científica:** 2.192 registros diarios NASA POWER, punto 4,2° N / 75,0° O, Tolima, 2020–2025, consultados el 7 de octubre de 2026. Se conserva el JSON original, la URL exacta y su SHA-256. NASA POWER distribuye estimaciones provenientes de satélite y reanálisis. No son valores aleatorios generados por este proyecto, pero tampoco son sensores instalados en una finca.

| Campo descargado | Unidad según JSON | Uso |
|---|---|---|
| ALLSKY_SFC_SW_DWN | kWh/m²/día | Radiación diaria horizontal; no electricidad producida por paneles |
| PRECTOTCORR | mm/día | Precipitación diaria corregida |
| T2M | °C | Contexto en registro; no interviene en la red de tres variables |

**Calculado desde esos datos:** umbrales por medianas, conteos, CPT, posteriores, escenarios históricos y validación temporal.

**Diseño académico, no medición:** estructura de la red, definición de “apto”, suavizado, umbral de revisión, operadores STRIPS, costos en puntos, cobertura relativa y adversario Minimax. Los priors alternativos de sensibilidad son supuestos deliberados para probar estabilidad. No se atribuyen a NASA.

El resultado de éxito se define operacionalmente como acertar una ventana meteorológica con lluvia baja y radiación alta al día siguiente. No hay registros de caudal, humedad del suelo, averías, rendimiento agrícola ni éxito físico del riego. No se puede presentar el posterior como probabilidad de que una bomba funcione o de que el cultivo mejore.

## Preparación y evaluación sin fuga temporal

Se excluyen faltantes `-999` o no finitos; en esta descarga no hay faltantes. Solo se forman pares de días consecutivos. Las medianas se calculan con 2020–2024: radiación 4,6807 kWh/m²/día y lluvia 2,24 mm/día. El cuartil 75 de lluvia (6,44 mm/día) solo interviene en la utilidad.

Entrenamiento: 1.826 pares cuyo día siguiente aún pertenece a 2020–2024. Prueba: 364 pares de 2025. Se omite el par 2024-12-31 → 2025-01-01 para mantener ambos conjuntos separados, y el último día no tiene etiqueta futura disponible. Ningún resultado de 2025 ajusta las CPT.

Los escenarios del benchmark se eligen de forma determinista: primer día de prueba de cada perfil $(C,E)$, ordenados por posterior; se conservan mínimo, central y máximo. Son casos ilustrativos; la validación predictiva usa los 364 pares, no solo estos tres.

## Red bayesiana

- $C=1$: lluvia de hoy $\leq 2{,}24\;\mathrm{mm}/\text{día}$.
- $E=1$: radiación de hoy $\geq 4{,}6807\;\mathrm{kWh}/(\mathrm{m}^{2}\cdot\text{día})$.
- $R=1$: mañana cumple simultáneamente ambos criterios.

Red: $C\rightarrow E$, $C\rightarrow R$ y $E\rightarrow R$. Las flechas describen una factorización probabilística; no demuestran causalidad entre lluvia y radiación.

$$
P(C,E,R)=P(C)\,P(E\mid C)\,P(R\mid C,E).
$$

Cada CPT binaria se estima con suavizado de Laplace, con $\alpha=1$:

$$
\widehat{P}=\frac{\text{conteo favorable}+1}{\text{conteo del contexto}+2}.
$$

El suavizado evita probabilidades cero; es una elección metodológica, no un dato adicional. `results/cpt_counts.csv` contiene todos los conteos y factores.

| $C$ | $E$ | $n(C,E)$ | $n(R=1,C,E)$ | $P(R=1\mid C,E)$ |
|---|---|---:|---:|---:|
| 0 | 0 | 571 | 77 | 13,6126% |
| 0 | 1 | 340 | 65 | 19,2982% |
| 1 | 0 | 342 | 132 | 38,6628% |
| 1 | 1 | 573 | 299 | 52,1739% |

Ejemplo de probabilidad conjunta:

$$
\begin{aligned}
P(C=1,E=0,R=1)
&=P(C=1)\,P(E=0\mid C=1)\,P(R=1\mid C=1,E=0)\\
&\approx 0{,}501094\times 0{,}374046\times 0{,}386628\\
&\approx 0{,}0724665\approx 7{,}2466\%.
\end{aligned}
$$

Caso de demo 2025-01-24: lluvia 0,09; radiación 3,5616; $C=1$, $E=0$.

$$
\begin{aligned}
P(R=1)&\approx 31{,}4625\%,\\
P(R=1\mid C=1,E=0)&\approx 38{,}6628\%.
\end{aligned}
$$

La evidencia aumenta la expectativa 7,20 puntos porcentuales; no garantiza el evento. De hecho, la etiqueta real del día siguiente en este caso es $R=0$.

En 2025: Brier 0,17299 frente a 0,19881 del predictor constante basado en el prior; menor es mejor. Exactitud con corte 0,5: 73,35%. La referencia “siempre $R=0$” logra 73,08%, porque $R=1$ solo ocurre el 26,92% de los días de prueba. Por eso no se usa la exactitud como única prueba de calidad. No hay validación en otros lugares o años futuros ni garantía de calibración.

## STRIPS: buscar, no escribir una ruta fija

Estado inicial académico:

$$
S_0=\{\text{base},\text{kit},\text{pendiente}\}.
$$

Para cada alternativa se construye una meta distinta: `solar_programado`, `respaldo_programado` o `espera_registrada`. Si $p<0{,}60$, donde $p$ es el posterior, se agrega `sensores_revisados`. El umbral 0,60 es una política ilustrativa; revisar sensores no modifica automáticamente la evidencia meteorológica.

Los operadores disponibles tienen PRE, ADD y DELETE en `OPS`. BFS explora estados y encuentra un plan corto; no se almacena una secuencia predeterminada como solución. Para evitar metas incompatibles, cada búsqueda permite solo el operador terminal de su alternativa. Los tres planes se validan volviendo a ejecutar sus precondiciones y efectos. “Inspeccionar paneles” está disponible, pero no ayuda a estas metas.

En el caso base: ir a parcela → verificar equipo → revisar sensores → programar solar. El grafo presenta la unión de los tres planes candidatos; no pretende mostrar todos los estados que BFS exploró.

## Minimax y utilidad

MAX es el agente. MIN representa un ambiente adverso como prueba de estrés, no una persona ni una distribución de probabilidad. Niveles: MAX elige solar/respaldo/posponer → MIN elige ambiente → MAX mantiene/refuerza → MIN elige ambiente de la segunda ventana. Los días de estrés son registros completos cercanos a los cuantiles 10, 50 y 90 de radiación de entrenamiento: 2022-11-20, 2021-03-29 y 2023-06-13. Se preserva lluvia y radiación de la misma fecha. Combinar ventanas genera escenarios contrafactuales; no son una secuencia histórica observada ni un pronóstico.

La respuesta “reforzar” es una opción contingente del árbol futuro. STRIPS prepara la decisión inmediata. La rama principal del caso base elige mantener; no hay un refuerzo omitido en su plan inmediato. El agente tendría que observar de nuevo y replanificar antes de una acción futura.

Para cada ventana, $H$ es la radiación y $L$ es la lluvia del registro histórico. $\operatorname{mediana}(H)$ y $Q_{0{,}75}(L)$ se calculan con entrenamiento:

$$
\begin{aligned}
s&=\min\left(1,\frac{H}{\operatorname{mediana}(H)}\right),\\
d&=\max\left(0,1-\frac{L}{Q_{0{,}75}(L)}\right),\\
w&=0{,}5+0{,}5d.
\end{aligned}
$$

La cobertura relativa $q$ depende de la alternativa $a$:

$$
q(a)=
\begin{cases}
s, & a=\text{solar},\\
\min(1,s+0{,}35), & a=\text{respaldo},\\
0, & a=\text{posponer}.
\end{cases}
$$

Para las alternativas activas, reforzar aplica $q\leftarrow\min(1,q+0{,}20)$; al posponer, la cobertura permanece en $0$.

La utilidad usa el posterior $p$ y el promedio de las dos ventanas evaluadas:

$$
U=p\cdot\frac{1}{2}\sum_{j=1}^{2}
\left[w_j\left(100q_j-40(1-q_j)\right)\right]
-\operatorname{costo}.
$$

Costos base en puntos:

$$
\operatorname{costo}_{\text{base}}(a)=
\begin{cases}
8, & a=\text{solar},\\
28, & a=\text{respaldo},\\
0, & a=\text{posponer}.
\end{cases}
$$

Reforzar suma $12$ puntos al costo base. **Son puntos de preferencia**, no pesos colombianos ni costos de una finca. Tampoco $q$ es eficiencia fotovoltaica, agua bombeada o energía medida. La fórmula combina datos ambientales reales con preferencias explícitas para comparar alternativas; cambiar esos pesos puede cambiar la decisión. El posterior no es una probabilidad de acción causal y se usa como peso común de contexto.

### Comparación solicitada

Se evalúan $3\times3\times3=27$ combinaciones: tres escenarios, profundidades $\{2,3,4\}$ y tres variantes (naive, poda $\alpha$–$\beta$ y heurística). La heurística usa un horizonte una capa más corto ($d_{\text{efectiva}}=d_{\text{solicitada}}-1$, campo `effective_depth`) y completa ambientes no observados con el registro central. Por tanto, su ahorro incluye buscar menos; no se afirma una comparación de igual horizonte con el método exacto. A profundidad 4, naive y alpha-beta llegan a hojas completas y son comparables exactamente.

En los 27 casos la decisión es solar. Alpha-beta preserva decisión y valor del naive a igual profundidad. A profundidad 4 pasa de 31 a 17 nodos internos expandidos (45,2% menos) en estos casos. No confundir expandidos con visitados: el árbol completo tiene 67 nodos y 36 hojas.

La heurística no cambia el movimiento en estos tres escenarios. El arrepentimiento se calcula respecto a la mejor utilidad Minimax completa a profundidad 4, reevaluando la acción escogida a ese horizonte. Es cero en estos casos; eso no prueba que siempre sea cero. El CSV registra cada valor y tiempo real del equipo de ejecución.

## Sensibilidad con evidencia fija

Se varía el prior $\pi=P(R=1)$ entre $0{,}05$, $0{,}20$, el prior ajustado, $0{,}40$, $0{,}60$ y $0{,}80$. Se mantiene $P(C,E\mid R)$, calculado desde la conjunta ajustada.

Con la misma evidencia $C=1$, $E=0$, se define:

$$
L_r=P(C=1,E=0\mid R=r),\qquad r\in\{0,1\}.
$$

El posterior para cada prior alternativo se calcula como:

$$
P_{\pi}(R=1\mid C=1,E=0)
=\frac{L_1\pi}{L_1\pi+L_0(1-\pi)}.
$$

No se presupone independencia de $C$ y $E$ dado $R$.

| Prior | Posterior | Decisión | Pasos |
|---:|---:|---|---:|
| 5% | 6,74% | Posponer | 3 |
| 20% | 25,56% | Solar | 4 |
| 31,46% (ajustado) | 38,66% | Solar | 4 |
| 40% | 47,79% | Solar | 4 |
| 60% | 67,32% | Solar | 3 |
| 80% | 84,60% | Solar | 3 |

Con prior 5% la meta es espera + sensores revisados. Con 20% se programa solar, conservando la revisión. Con 60% ya no se agrega esa revisión. Esto muestra una decisión estable en parte del rango y sensible cerca de los extremos y del umbral de política.

## Flujo y entregables

Observar registro → Bayes calcula posterior → STRIPS prepara planes válidos → Minimax decide → seleccionar el plan correspondiente → registrar recomendación. No se acciona ningún equipo físico.

- `agent.py`: implementación comentada; `test_agent.py`: comprobaciones relevantes.
- `data/`: JSON original, CSV diarios, pares, URL y `provenance.json`.
- `results/minmax_tree.png`: árbol completo con valores y rama destacada.
- `results/strips_graph.png`: estados, planes y leyenda.
- `results/bayes_bars.png`: posterior y evidencia.
- `results/agent_summary.png`: síntesis coherente del plan, posterior y decisión.
- `results/agent_log.txt`: mismo flujo impreso en consola.
- `results/minimax_benchmark.csv`, `sensitivity.csv`, `cpt_counts.csv`, `test_predictions.csv`, `report.json`: auditoría reproducible.
- `guion_demo.md`: explicación y demo de cinco minutos.

## Fuentes y referencia

1. NASA Langley Research Center, POWER Project. (s. f.). *Daily API*. https://power.larc.nasa.gov/docs/services/api/temporal/daily/ (consulta: 7 de octubre de 2026).
2. NASA Langley Research Center, POWER Project. (s. f.). *Data Sources*. https://power.larc.nasa.gov/docs/methodology/data/sources/ (consulta: 7 de octubre de 2026). Documenta CERES SYN1deg y MERRA-2 y sus resoluciones de origen.
3. NASA POWER. (2026). *Serie diaria 2020–2025, punto 4,2° N / 75,0° O*. Dataset descargado vía API; consulta exacta en `data/request_url.txt`, respuesta preservada en `data/nasa_power_raw.json`, metadatos y hash en `data/provenance.json`.
4. Material de clase suministrado: `ParcialCorte2 IA(1).pdf` (requisitos) y `Parcial Inteligencia Artificial(1).pdf` (referencia bancaria). Orientan la entrega, no son la fuente de nuestros valores.

Reconocimiento: los datos fueron obtenidos del proyecto NASA POWER del NASA Langley Research Center, financiado por NASA Earth Science/Applied Science. Cálculos, discretización, hipótesis de decisión y software: elaboración del proyecto.
