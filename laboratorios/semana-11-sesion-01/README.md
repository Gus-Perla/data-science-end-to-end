# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

Hay que versionar el **código**, los **datos** y el **modelo** (con sus parámetros y métricas), porque un resultado solo se puede reproducir si se tienen los tres en la versión exacta que lo produjo. Las herramientas habituales para cada uno son **Git** para el código, **DVC** (*Data Version Control*) para los datos, que guarda en Git un archivo pequeño con el hash de cada archivo y deja el contenido en un almacenamiento aparte, y **MLflow** para los modelos y los experimentos: registra parámetros, métricas y artefactos de cada corrida.

**¿Por qué no alcanza con Git para versionar los datos?**

Porque Git está hecho para archivos de texto pequeños que cambian línea por línea. Los datos suelen pesar cientos de MB o GB, a veces son binarios (Parquet, imágenes), y Git guarda una copia completa de cada versión en el historial: el repositorio se vuelve enorme y lento, y servicios como GitHub rechazan archivos de más de 100 MB. Además, los datos a menudo viven fuera del repositorio (un bucket, un warehouse) y cambian sin que nadie haga commit. Lo que sí se puede poner en Git es una **referencia** a la versión de los datos (un hash o huella), que es lo que hacen DVC y la huella que registramos en MLflow.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Todo lo que determina el número: la versión del código y de la preparación, la versión exacta de los datos (una huella del archivo), cómo se dividieron entrenamiento y prueba (la semilla o la fecha de corte), los hiperparámetros del modelo y su semilla, las versiones de las librerías, y qué métrica se calculó sobre qué conjunto. Si cada corrida registra eso, se comparan los registros y la diferencia aparece sola.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ninguno "tiene razón": los dos números son correctos para lo que midieron. El nuestro da **1.2754** en prueba y el del compañero **1.1867**. La salida dice el nombre del modelo, la división (42 contra 7) y los dos errores, así que permite ver que **se midieron sobre conjuntos de prueba distintos**. Lo que no permite saber es si todo lo demás era igual: los hiperparámetros, la versión del CSV o cuántas filas usó cada uno; y al reiniciar el kernel esa salida desaparece. Lo que hay que decir no es cuál de los dos es correcto, sino que los dos números no son comparables.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La **división de los datos**: el modelo, sus hiperparámetros y el archivo son los mismos, pero el compañero dividió con `random_state=7`, así que su modelo aprendió con otras 2380 filas y se evaluó con otras 595 (con 595 filas, la suerte de qué viajes caen en prueba mueve el MAE en décimas). En la tabla del paso 5 esperaba ver distinto **`params.semilla_division`** (42 contra 7), con la misma `huella_datos` (4800204d50c0) y el mismo `n_estimators` (100). Así fue.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Porque los parámetros y las métricas describen el modelo, pero no **son** el modelo. Guardar el pipeline entrenado (con la preparación incluida) permite volver a cargarlo para comprobar que reproduce la métrica registrada, usarlo para predecir sin reentrenarlo (servirlo en producción, que es lo que viene en la Semana 12), compararlo con otro sobre datos nuevos, o investigar después por qué se equivocó en algún caso. Reentrenar "con los mismos parámetros" no siempre da el mismo modelo (cambian librerías, datos o semillas), y algunos modelos tardan horas.

**Lo que mostró la tabla del paso 5:**

| corrida | semilla de la división | huella de los datos | n_estimators | MAE en prueba | segundos |
|---|---|---|---|---|---|
| gradient boosting (compañero) | 7 | 4800204d50c0 | 100 | 1.1867 | 0.45 |
| gradient boosting | 42 | 4800204d50c0 | 100 | 1.2754 | 0.42 |
| bosque aleatorio | 42 | 4800204d50c0 | 200 | 1.3016 | 3.32 |

Mismos datos y mismos hiperparámetros: lo único que cambia entre los dos gradient boosting es la semilla de la división, y eso solo ya mueve el error en 0.09.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La **red neuronal**, con MAE en prueba de **1.1695** (run `eb1cc2fc8b1d4d8baecdb4fb2c4995c4` en mi registro local). Le gana al segundo, el gradient boosting (1.2754), por **0.1058** dólares por viaje. Después vienen el bosque (1.3016), Ridge sin interacciones (2.5565) y la línea base (12.5308). La red, eso sí, tardó 46 s en entrenar contra 0.42 s del gradient boosting.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que el error reproducido coincida (1.1695052296 en los dos) garantiza que **el artefacto guardado es exactamente el modelo que produjo ese número**: el pipeline completo (preparación + red) quedó bien guardado, se carga sin depender del kernel, y X_prueba es la misma. No garantiza que el modelo **sea bueno en producción**: el número está medido sobre una sola división de 595 viajes (con otra semilla podría cambiar en décimas, como en el Ejercicio 1), sobre datos históricos que pueden no parecerse a los futuros, y la diferencia de 0.1 con el gradient boosting podría ser ruido. Tampoco dice nada de la latencia, del costo o de la explicabilidad.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por la **huella de los datos**: `params.huella_datos` es el hash MD5 del archivo, y cambiar un solo byte cambia la huella. Las corridas nuevas aparecerían con una huella distinta de `4800204d50c0`, aunque el nombre del modelo, los hiperparámetros y la semilla fueran iguales. Así se sabe que sus métricas no son comparables con las anteriores sin tener que abrir el CSV.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Que el registro guarda **todos** los hiperparámetros de la red, incluso los que no escribimos: `hidden_layer_sizes = (64,)`, `max_iter = 1000`, `solver = adam`, `activation = relu`, `alpha = 0.0001`, `learning_rate_init = 0.001`, `early_stopping = False`, `random_state = 42`, etc. En una tabla a mano solo anotaríamos "red neuronal, 64 neuronas", y si scikit-learn cambia un valor por defecto, no sabríamos qué se usó. También muestra el tiempo de entrenamiento (46 s contra menos de 4 s de los demás), la huella de los datos, la hora y el autor de la corrida, y el modelo guardado como artefacto, listo para cargar.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

_(tu respuesta)_

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Podría cambiar en décimas: el Ejercicio 1 lo mostró en el registro, porque el mismo gradient boosting dio 1.2754 con la división 42 y 1.1867 con la 7, una diferencia (0.09) casi tan grande como la que separa a la red del gradient boosting (0.11). Así que la elección del ganador puede depender de la suerte de la división. Para saberlo habría que medir la variación: validación cruzada (varias divisiones, reportando media y desviación) o un intervalo de confianza del error por *bootstrap* sobre el conjunto de prueba.
