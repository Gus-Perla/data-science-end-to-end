# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el script `pipeline_proyecto.py` de tu proyecto, de la Semana 9, Sesión 2.

Los laboratorios de hoy son notebooks: entrenar y comparar modelos es experimentar, y en un notebook vuelves a correr solo el modelo que cambiaste, sin reentrenar los demás. La preparación no se copia en el notebook: se importa del script, para que todos los modelos usen exactamente la misma.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

Un ensamblaje combina las predicciones de varios modelos (normalmente del mismo tipo, como árboles) para obtener una predicción mejor que la de cualquiera de ellos por separado.

- **Bagging** (*bootstrap aggregating*): entrena muchos modelos **en paralelo e independientes**, cada uno sobre una muestra con reemplazo de los datos (y, en el bosque aleatorio, con un subconjunto al azar de columnas en cada corte). La predicción final es el **promedio** (regresión) o el voto (clasificación). Reduce sobre todo la **varianza**.
- **Boosting**: entrena los modelos **en secuencia**; cada uno se enfoca en lo que los anteriores hicieron mal (en *gradient boosting*, se ajusta a los residuos del ensamblaje hasta ese momento). La predicción es una **suma ponderada** de modelos débiles (árboles poco profundos). Reduce sobre todo el **sesgo**.

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

Porque los errores de cada árbol son en parte aleatorios: cada uno se sobreajusta a su propia muestra de los datos y se equivoca de forma distinta. Al promediar, los errores que van en direcciones opuestas se cancelan y lo que queda es el patrón que todos comparten. El promedio de N errores poco correlacionados tiene una varianza mucho menor que la de cada uno. Funciona mientras los árboles sean razonablemente buenos y **diferentes entre sí**; por eso el bosque aleatorio usa muestras y columnas al azar, para que sus errores no estén correlacionados.

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

Se está **sobreajustando**: memorizó detalles y ruido de los datos de entrenamiento en lugar del patrón general, así que no generaliza. La brecha entre el error de prueba y el de entrenamiento mide cuánto. El Ejercicio 1 lo muestra: el árbol sin límite tiene un error de 0.0029 en entrenamiento y 1.7017 en prueba.

## Ejercicio 1: línea base, un árbol y un bosque

Notebook: [`starter/ejercicio1_linea_base_y_ensamblaje.ipynb`](./starter/ejercicio1_linea_base_y_ensamblaje.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. La preparación de la Semana 9 está en [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py) y el notebook la importa. Ya trae una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos, cada uno en su celda. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Esperaba un error grande, del orden de la dispersión de la tarifa: la mediana de entrenamiento es 22.53 y la tarifa va de unos 5 a 70, así que en promedio se equivoca por más de 10. Fue **13.20 en entrenamiento y 12.53 en prueba**. Sirve como **piso de comparación**: cualquier modelo que use columnas tiene que ganarle por bastante para justificar su complejidad. Si un modelo sofisticado apenas le gana, algo anda mal (los datos no tienen señal, o hay un error en el pipeline). Además, da la escala del problema: un MAE de 1.3 solo se entiende comparado con 12.5.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, un error casi 0: cada hoja termina con un viaje (o con viajes idénticos), así que predice exactamente su tarifa. Fue **0.0029** (no exactamente 0 porque hay viajes con las mismas features y tarifas distintas que caen en la misma hoja). En prueba, bastante peor, porque memorizó el ruido de cada viaje: **1.7017**, la mayor brecha de la tabla (1.70).

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, la brecha baja: el árbol no puede memorizar cada viaje, sube un poco el error de entrenamiento y baja el de prueba. Así fue: con `max_depth=8` la brecha bajó de 1.70 a **0.26** y el error en prueba de 1.70 a **1.27**. Con 200 árboles promediados esperaba que la prueba mejorara respecto al árbol sin límite y que la brecha bajara, aunque los árboles del bosque no tienen límite y siguen ajustando mucho el entrenamiento. El bosque bajó la prueba a **1.30** (brecha 0.83), pero no le ganó al árbol con `max_depth=8`: en este dataset la señal es casi lineal en la distancia y el ruido es lo que queda; limitar la profundidad fue tan efectivo como promediar, y 100 veces más rápido (0.04 s contra 4.5 s).

**Lo que confirmó la tabla, TODO por TODO:**

| modelo | MAE entrenamiento | MAE prueba | brecha | segundos |
|---|---|---|---|---|
| línea base (mediana) | 13.1998 | 12.5308 | -0.6690 | 0.03 |
| árbol sin límite | 0.0029 | 1.7017 | 1.6988 | 0.05 |
| árbol `max_depth=8` | 1.0098 | 1.2728 | 0.2630 | 0.04 |
| bosque aleatorio (200) | 0.4739 | 1.3016 | 0.8276 | 4.48 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Notebook: [`starter/ejercicio2_regularizacion.ipynb`](./starter/ejercicio2_regularizacion.ipynb), sobre el mismo `viajes_features.csv` y con la misma preparación importada. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe **3275 columnas** y hay solo **2380 filas** de entrenamiento: más coeficientes que ecuaciones. La regresión lineal puede ajustar el entrenamiento casi perfectamente (0.8953) combinando columnas que solo existen para unos pocos viajes, y los coeficientes se vuelven enormes y compensados entre sí. En prueba, cualquier combinación ligeramente distinta multiplica esos coeficientes gigantes y el error explota: **3,401,788,495.57**, una brecha de más de tres mil millones. Es el sobreajuste llevado al extremo.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento, el error **subió** de 0.8953 a **1.0272**; en prueba, **bajó** de 3.4 mil millones a **1.3209**. Ridge agrega una penalización a la suma de los cuadrados de los coeficientes (`alpha = 10`), así que ya no puede usar coeficientes gigantes para ajustar cada viaje de entrenamiento: renuncia a un poco de ajuste en los datos que ve a cambio de coeficientes pequeños y estables, que generalizan. Ese poco de ajuste perdido es el error de entrenamiento extra.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

**Gradient boosting**, con un MAE en prueba de **1.2754** (entrenamiento 1.0705, brecha 0.20, la más baja de los modelos que aprenden). La línea base tiene 12.5308, así que se equivoca **11.26 menos por viaje, un 90% menos** (casi 10 veces menos error). Ridge grado 3 queda cerca (1.3209), pero tarda más y usa 3275 columnas.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Porque la brecha no mide solo el modelo, también la diferencia entre los dos conjuntos. La línea base predice 22.53 para todos; su error depende solo de qué tan dispersas están las tarifas alrededor de ese número. Por azar de la división, el conjunto de prueba tiene tarifas un poco menos dispersas: desviación estándar 17.32 contra 18.17 en entrenamiento, y menos viajes de tarifa fija de 70 (8.9% contra 10.4%), que son los que más lejos quedan de la mediana. Con 595 filas de prueba, una diferencia así es ruido de muestreo; por eso una brecha pequeña, positiva o negativa, no significa nada por sí sola.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el notebook tú.

### Entregable 1: `modelos_proyecto.ipynb`

Un notebook, en la misma carpeta que tu `pipeline_proyecto.py` (`psets/proyecto/modelado/`), que importe de ahí la preparación y la división (`from pipeline_proyecto import cargar_y_dividir, construir_preparacion`) en vez de copiarlas, y que entrene:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento, escrita por el notebook. Antes de hacer commit, reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa, no de celdas corridas en momentos distintos.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

_(tu respuesta)_

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

_(tu respuesta)_

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

_(tu respuesta)_

**¿Cuánto mejora tu mejor modelo a la línea base?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

Además del error: el **costo** por predicción (un ensamblaje corre gratis en una CPU; un LLM cobra por token en cada viaje), la **latencia** (la app necesita el precio en milisegundos al pedir el taxi; una llamada a un LLM tarda segundos), la **consistencia** (el mismo viaje debe dar el mismo precio; un LLM puede responder distinto cada vez o devolver texto en lugar de un número), el **tiempo de entrenamiento y mantenimiento** (reentrenar el ensamblaje es barato; la red neuronal necesita más ajuste), la **explicabilidad** (poder justificarle a un cliente o a un regulador por qué cuesta eso) y la **privacidad** (enviar datos de viajes a un tercero). Para un problema tabular con 3000 filas, el ensamblaje probablemente gana en casi todo.
