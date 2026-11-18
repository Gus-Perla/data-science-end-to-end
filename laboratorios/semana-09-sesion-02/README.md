# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

scikit-learn ya viene en el Codespace. En tu laptop, instálalo con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. El `requirements.txt` del repositorio ya fija una versión posterior.

Ten a mano tu notebook de features de la sesión pasada. Los tres ejercicios de hoy son scripts, no notebooks: un pipeline de preparación es código de producción, que tiene que correr igual cada vez y que otros archivos importan. En el Ejercicio 3, la preparación de tu proyecto pasa del notebook donde la probaste a ese script.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

La fuga de información (*data leakage*) ocurre cuando el modelo, durante el entrenamiento, tiene acceso a información que no tendrá en el momento real de predecir: datos del conjunto de prueba, o columnas que contienen (directa o indirectamente) la respuesta. El modelo aprende a apoyarse en esa información, y la evaluación lo premia porque el conjunto de prueba la contiene igual. Por eso las métricas salen excelentes. En producción esa información no existe (la columna no está, o los datos nuevos no se parecen a los que se usaron para preparar), y el rendimiento cae: la evaluación medía algo que el modelo nunca va a poder hacer de verdad.

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

La **fuga del objetivo** (*target leakage*): usar como feature una variable que solo se conoce después del evento que se quiere predecir. La duración del viaje se conoce al terminar, igual que la tarifa, y está casi determinada junto con ella (correlación 0.956). La otra forma, la **contaminación entre entrenamiento y prueba** (calcular estadísticas de preprocesamiento con todos los datos antes de dividir), es la del Ejercicio 1 de hoy.

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de entrenamiento. Si la media y la desviación se calculan con todos los viajes, el preprocesamiento ya "vio" los viajes de prueba y la evaluación deja de medir el rendimiento sobre datos nuevos. En producción, el escalador tendrá que usar estadísticas calculadas con el pasado, nunca con los datos que llegan. Con la distancia el cambio numérico es pequeño (la media pasa de 8.5034 con todo a 8.5576 solo con entrenamiento), pero la regla es la misma para cualquier paso que aprende algo de los datos: imputar, escalar, codificar o seleccionar features se ajusta (`fit`) solo con entrenamiento.

## Ejercicio 1: encontrar y corregir la fuga

Archivo: [`starter/ejercicio1_fuga.py`](./starter/ejercicio1_fuga.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv): los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. La primera parte del script es el código de un compañero: imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. La segunda parte es la versión correcta, con 3 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-02/starter
python3 ejercicio1_fuga.py
```

**Respuesta 1, antes de ver la solución:** en la versión del compañero, ¿en qué línea entra el conjunto de prueba al entrenamiento? ¿Qué número de la salida lo delata?

En la **línea 26**: `X_preparado = escalador.fit_transform(imputador.fit_transform(X))`. El imputador y el escalador aprenden la mediana, la media y la desviación con las 2975 filas, antes de `train_test_split`, así que las filas que luego serán de prueba ya influyeron en la preparación. Lo delata la salida `Filas que vio el escalador: 2975`: si solo hubiera aprendido de entrenamiento (80%), serían 2380.

**Respuesta 2 (TODO 1 y 3):** sin resolver, la versión correcta dice que tiene 2975 filas de prueba. ¿Qué está midiendo ese error, si prueba y entrenamiento son las mismas filas?

Mide el **error de entrenamiento**, no el de generalización: el modelo se evalúa sobre las mismas 2975 filas con las que aprendió. Dice qué tan bien se ajusta a datos que ya vio, no qué tan bien predice viajes nuevos. Para una regresión lineal con dos variables la diferencia es pequeña (2.5590), pero con un modelo flexible podría ser casi 0 y no significar nada.

**Respuesta 3:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Casi nada, y así fue: **2.5375 con fuga contra 2.5373 corregida**. Que casi no cambie **no significa que no había fuga**: la fuga estaba (el escalador vio 2975 filas y usó media 8.5034 en lugar de 8.5576), solo que en este caso es benigna porque la mediana y la media de 2975 viajes casi no cambian al quitar el 20%, y una regresión lineal es invariante al escalado. El tamaño del efecto depende de los datos y del paso que filtra: con pocos datos, con *target encoding*, con selección de features o con datos que cambian en el tiempo, la misma fuga puede inflar mucho la métrica. Lo que se corrige es el procedimiento, no un número.

**Lo que confirmó la salida, TODO por TODO:**

| versión | filas que vio el escalador | media de distancia | filas de prueba | MAE en prueba |
|---|---|---|---|---|
| compañero (con fuga) | 2975 | 8.5034 | 595 | 2.5375 |
| sin resolver | (no escala) | — | 2975 | 2.5590 |
| correcta (TODO 1 a 3) | 2380 | 8.5576 | 595 | 2.5373 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Archivo: [`starter/ejercicio2_pipeline.py`](./starter/ejercicio2_pipeline.py), sobre el mismo `viajes_features.csv`. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_pipeline.py
```

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan **25 columnas** (2380 × 25 en entrenamiento y 595 × 25 en prueba):

- Numéricas: 2 (`distancia_km`, `pasajeros`), imputadas y escaladas, siguen siendo 2.
- Categóricas: one-hot crea una columna por categoría vista en entrenamiento: 8 de `zona_origen` + 8 de `zona_destino` + 2 de `metodo_pago` = 18.
- Ya listas: 5 (`hora_sin`, `hora_cos`, `es_fin_de_semana`, `es_hora_pico`, `es_tarifa_fija`) pasan sin cambios.

2 + 18 + 5 = 25.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de **entrenamiento**: el imputador aprendió `distancia_km = 6.94` y `pasajeros = 1.0`, iguales a las medianas de entrenamiento; la de todo el dataset es 6.93. Demuestra que el `fit` del `ColumnTransformer` solo usó las filas de entrenamiento: el conjunto de prueba no participó en lo que aprendió la preparación, así que no hay fuga. (La diferencia es pequeña, pero es la que distingue un procedimiento correcto de uno con fuga.)

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

El Bronx no apareció en entrenamiento, así que con `handle_unknown="ignore"` el viaje queda con **las 8 columnas de `zona_origen` en 0** (`zona_origen_en_1 = 0`): el modelo lo trata como "ninguna zona conocida" y predice con el resto de las features. El viaje sigue teniendo **25 columnas**, las mismas que espera el modelo. Con `pd.get_dummies` aplicado solo a ese viaje, saldrían columnas como `zona_origen_Bronx`, `zona_destino_Midtown` y `metodo_pago_tarjeta`: una sola columna por variable, con nombres y cantidad distintos a los del entrenamiento. El modelo recibiría una matriz de otra forma y fallaría (o, si se alinean por posición, usaría columnas equivocadas). Por eso la codificación tiene que aprenderse una vez, con entrenamiento, y reutilizarse.

## Ejercicio 3: el pipeline de tu proyecto, como script

Copia la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py) a `psets/proyecto/modelado/`, al lado de tu notebook de features. Desde la próxima sesión, tus notebooks de modelos importan este archivo en vez de copiar su código:

```python
from pipeline_proyecto import cargar_y_dividir, construir_preparacion
```

### Paso 1: pasa tus features a `agregar_features` (5 minutos)

Copia de tu notebook de la sesión pasada solo las líneas que construyen tus 2 features, dentro de la función `agregar_features`. Sin gráficos, sin `print` y sin las celdas que usaste para probar ideas: esas se quedan en el notebook, como evidencia de por qué elegiste cada feature.

### Paso 2: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

_(tu clasificación)_

### Paso 3: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha dentro de `cargar_y_dividir`; si no, al azar. Justifica.

_(tu decisión)_

### Paso 4: corre el script (10 minutos)

Desde una terminal, en `psets/proyecto/modelado/`:

```
python3 pipeline_proyecto.py
```

Si corre de principio a fin en una terminal nueva, sin el notebook abierto, otro código ya lo puede importar. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

_(lo que imprimió tu script)_

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Está **sobreajustando** (*overfitting*): aprendió detalles y ruido propios del conjunto de entrenamiento en lugar del patrón general, así que no generaliza a datos nuevos. Es un problema de alta varianza: el modelo es demasiado flexible para la cantidad de datos (por ejemplo, un árbol muy profundo). Se corrige limitando su complejidad (profundidad máxima, regularización), con más datos o combinando muchos modelos en un ensamble (como *bagging* o un bosque aleatorio), que promedia la varianza de cada uno. El modelo que hay que elegir es el de menor error en **prueba**, no en entrenamiento.
