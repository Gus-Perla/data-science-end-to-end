# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada, y tu `pipeline_proyecto.py`.

Como la sesión pasada, los laboratorios son notebooks que importan la preparación de un script (`preparacion_taxis.py` en los Ejercicios 1 y 2, tu `pipeline_proyecto.py` en el 3).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

- **Aprendizaje automático clásico:** para datos **estructurados y tabulares**, incluso con conjuntos pequeños, cuando hay una etiqueta bien definida que predecir. El artículo dice que el ML clásico "gana en problemas de datos tabulares y estructurados": mantenimiento predictivo con series de sensores, modelos de riesgo, detección de fraude o diagnóstico a partir de valores de laboratorio, donde importan la auditabilidad y la velocidad.
- **LLM:** para datos **no estructurados, sobre todo texto**: generar texto, entender lenguaje con matices, atención al cliente, análisis de documentos, resumir notas clínicas. Conviene empezar con un LLM cuando la entrada es texto, la tarea requiere razonar sobre contexto o no hay datos etiquetados para entrenar.

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

La de **datos**. Mi proyecto predice fallas de discos SSD a partir de una tabla de atributos SMART y de ubicación (modelo, aplicación, rack, nodo): datos tabulares, numéricos y con una etiqueta clara (`failure`), justo el caso en el que el artículo dice que gana el ML clásico. Un LLM no tiene ventaja con números de sensores y no puede ver los millones de filas como contexto. Además, en mantenimiento predictivo se corre sobre muchos discos cada día, así que el costo y la latencia de un LLM se multiplican sin mejorar la precisión, como dice el artículo.

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error en prueba:

- **Costo**: de entrenar y, sobre todo, de cada predicción (los modelos propios corren gratis en una CPU; el LLM cobra por token en cada viaje).
- **Latencia**: la app tiene que mostrar el precio en milisegundos al pedir el taxi.
- **Consistencia**: el mismo viaje tiene que dar siempre el mismo precio, y siempre un número válido.
- **Explicabilidad**: poder justificarle al conductor o al pasajero por qué la tarifa es esa.
- **Tiempo de entrenamiento y mantenimiento**: cuánto cuesta reentrenar cuando cambian los datos.
- **Privacidad**: si los datos de los viajes salen hacia un tercero.

## Ejercicio 1: una red neuronal frente al ensamblaje

Notebook: [`starter/ejercicio1_red_neuronal.ipynb`](./starter/ejercicio1_red_neuronal.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). Ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división, importadas de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Significa que el optimizador llegó al máximo de iteraciones (200, el valor por defecto) **sin converger**: la pérdida todavía estaba bajando cuando se detuvo, así que la red quedó a medio entrenar. No es que la red sea mala, es que no terminó de aprender: con 200 iteraciones tenía un MAE de 1.60 en entrenamiento y 1.69 en prueba, peor que el gradient boosting (1.28). Al dejarla entrenar hasta 1000 iteraciones esperaba que convergiera y bajara el error, y así fue: paró sola en la **iteración 692** (sin advertencia) con MAE de **1.0882 en entrenamiento y 1.1695 en prueba**, mejor que el ensamblaje.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

**1729 números.** Entre la entrada y la capa oculta hay una matriz de 25 × 64 = 1600 pesos más 64 sesgos; entre la capa oculta y la salida, 64 × 1 = 64 pesos más 1 sesgo. 1600 + 64 + 64 + 1 = 1729. La salida lo confirma: `coefs_` tiene formas (25, 64) y (64, 1).

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La **distancia**: la exploración ya mostró una correlación de 0.96 con la tarifa. Así fue: `distancia_km` tiene una importancia de **0.9824**, seguida de lejos por `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028). Para explicar una tarifa de 70 dólares, el **ensamblaje** permite responder: sus importancias (y los cortes de sus árboles) muestran que la segunda variable que más usa es `es_tarifa_fija`, así que se puede decir "es un viaje entre JFK y Manhattan, que tiene tarifa fija". Con la red, los 1729 pesos no tienen un significado que se pueda leer; habría que usar herramientas aparte (como SHAP) para aproximar una explicación.

**Lo que confirmó la salida, TODO por TODO:**

| modelo | MAE entrenamiento | MAE prueba | segundos de entrenamiento |
|---|---|---|---|
| gradient boosting | 1.0705 | 1.2754 | 0.6 |
| red neuronal (200 iteraciones, sin converger) | 1.6009 | 1.6854 | — |
| red neuronal (TODO 1, convergió en 692 iteraciones) | 1.0882 | 1.1695 | 50 a 84 |

- **TODO 2:** la red aprendió **1729 parámetros**.
- **TODO 3:** las 3 columnas más importantes del ensamblaje son `distancia_km` (0.9824), `es_tarifa_fija` (0.0120) y `es_hora_pico` (0.0028).

La red gana por 0.11 de MAE en prueba, pero tarda unas 100 veces más en entrenar y no se puede explicar directamente.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Notebook: [`starter/ejercicio2_tres_soluciones.ipynb`](./starter/ejercicio2_tres_soluciones.ipynb), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre las celdas hasta la que escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y corre sus celdas y la de la comprobación hasta que todas digan `OK`. Los modelos ya están entrenados en el kernel: no hace falta volver a correr las celdas de arriba. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

LLM usado: _(confirma qué LLM usaste; el costo está calculado con 2.50 USD por millón de tokens de entrada, ajústalo al precio de tu modelo)_. El prompt tiene 2902 caracteres (unos 726 tokens) y cubre 20 viajes, así que 10 000 viajes son 500 prompts.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.8957 | ~0.6 s de CPU, prácticamente 0 | prácticamente 0 (corre en milisegundos en el servidor) | Sí: importancias por columna (distancia 0.98, tarifa fija 0.012) y cortes de los árboles |
| Red neuronal | 0.9637 | 50 a 84 s de CPU, prácticamente 0 | prácticamente 0 | No directamente: 1729 pesos sin significado; requiere herramientas aparte (SHAP) |
| LLM | 1.0735 | 0 (no se entrena, pero el entrenamiento previo del proveedor es enorme) | ~0.91 USD al mes (500 prompts × 726 tokens × 2.50 USD/M), más la latencia de segundos por llamada | No de forma confiable: puede dar una justificación en texto, pero no es el cálculo real y cambia entre corridas |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

En general no: un LLM de chat genera con muestreo aleatorio (temperatura mayor que 0), así que en una conversación nueva los números cambian (los de 70 tienden a repetirse porque el patrón es obvio, pero los demás varían unos centavos o dólares). Para una app que cobra, eso es un problema serio: el mismo viaje, pedido dos veces, tendría precios distintos, y el resultado no se puede reproducir ni auditar si un pasajero reclama. Además, la respuesta puede venir con otro formato o con menos números. Un modelo entrenado da siempre la misma predicción para la misma entrada.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor es el viaje **de JFK Airport a JFK Airport** (1.01 km, 18 h, efectivo): tarifa real **11.46**, el LLM dijo **7.65** (error 3.81). Lo particular es que es un viaje corto dentro del aeropuerto, que no se parece a ninguno de los 30 ejemplos: el LLM lo estimó como un viaje corto cualquiera dentro de Manhattan (los ejemplos de ~1 km cuestan 7). Pero los viajes dentro de JFK cuestan más que la distancia sugiere: en entrenamiento hay **7 viajes JFK–JFK**, con tarifa media de 15.24 (de 12.39 a 18.67). El ensamblaje, que sí vio las 2380 filas, predijo 12.74 (error 1.28) y la red 13.79: aprendieron que el origen JFK sube el precio. El LLM solo ve los 30 ejemplos del prompt.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque el LLM se usa a mano: habría que pegar prompts con los 595 viajes (30 prompts como este, o uno muy largo que puede pasar el límite de contexto y degradar las respuestas), copiar y alinear 595 números, y pagar por todos esos tokens. Con 20 viajes la comparación es **poco confiable**: un solo viaje raro (como el JFK–JFK) mueve el MAE del LLM de 0.93 a 1.07, y en estos 20 hay 3 viajes de tarifa fija en los que todos aciertan 70. Los dos modelos dan 0.90 y 0.96 en la muestra, pero 1.28 y 1.17 en los 595: hasta el orden entre ellos se invierte. Sirve como señal de que el LLM no es descabellado, no para decidir entre opciones que difieren en décimas.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí tu preparación y tu división, y entrene el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

_(tu tabla)_

_(tu decisión)_

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Todo lo que define la corrida: la **versión del código** (el commit, incluida la preparación), la **versión de los datos** (qué archivo y qué filas, por ejemplo con un hash), la **división** entre entrenamiento y prueba (y su `random_state`), los **hiperparámetros** del modelo y su semilla, las **versiones de las librerías** (scikit-learn puede cambiar valores por defecto) y la **métrica** exacta y sobre qué conjunto se midió. Si eso queda registrado para cada corrida (en MLflow, por ejemplo), basta compararlas para encontrar la diferencia.
