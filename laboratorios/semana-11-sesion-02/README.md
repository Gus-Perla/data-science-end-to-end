# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Es cuando un modelo tiene una exactitud (*accuracy*) alta y aun así es inútil, porque la clase que importa es poco frecuente. Si solo el 5% de los casos son positivos, un modelo que siempre dice "negativo" acierta el 95% sin detectar ni un solo caso. La exactitud premia acertar la clase mayoritaria y esconde el fracaso en la minoritaria. En su lugar se recomiendan métricas que miran la clase de interés: **precisión** (de lo que marqué como positivo, cuánto lo era), **recall** o sensibilidad (de los positivos reales, cuántos encontré), **F1** (su media armónica), la **matriz de confusión** completa y curvas como la de precisión-recall, idealmente traducidas al costo real de cada tipo de error.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

El ejemplo típico es el de un indicador que se vuelve meta y la gente lo optimiza en lugar de lo que representaba: si a un equipo de soporte se lo evalúa por "tickets cerrados", empieza a cerrar tickets sin resolverlos; la métrica sube y el servicio empeora. Con los modelos pasa lo mismo: si elijo un modelo solo porque maximiza una métrica (accuracy, o el MAE sobre un único conjunto de prueba), termino optimizando la métrica y no el objetivo real. Puedo elegir un modelo que nunca detecta fraude porque tiene 95% de accuracy, o un modelo que ganó por décimas por la suerte de la división. Por eso hay que mirar varias métricas, su variación entre pliegues y el costo de negocio de cada error.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Podría cambiar en décimas. En el registro de la sesión pasada se vio: el mismo gradient boosting dio 1.2754 con la división 42 y 1.1867 con la división 7 del compañero, casi tanto como la diferencia entre la red y el gradient boosting (0.11). El registro guardaba la semilla de cada corrida, pero un solo número por corrida no dice cuánto varía: para eso hace falta validación cruzada, que es lo que hacemos hoy.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Esperaba que fueran parecidos, alrededor de 1.2 o 1.3, con variaciones de unas pocas centésimas (cada pliegue tiene 595 viajes, como la prueba de la Semana 10). Así fue: los 5 pliegues del gradient boosting dieron **1.28, 1.24, 1.16, 1.24 y 1.23** (promedio **1.2288**, desviación **0.0361**), y los del bosque **1.30, 1.27, 1.25, 1.34 y 1.28** (promedio 1.2880, desviación 0.0291). El 1.2754 de la Semana 10 estaba un poco por encima del promedio, pero dentro de la variación. El gradient boosting le gana al bosque por 0.06 en promedio, y en 4 de los 5 pliegues, así que esa diferencia sí parece real.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor. Con pliegues al azar, el modelo entrena con viajes de antes y de después de cada viaje de validación, así que ve condiciones parecidas (los mismos días, el mismo tráfico). En la temporal solo ve el pasado, como en producción, y si algo cambia con el tiempo el modelo no lo vio. Además, los primeros pliegues entrenan con pocos datos (el primero, con solo una sexta parte). Así fue: promedio **1.3099** contra 1.2288 al azar, con el doble de desviación (**0.0693**) y pliegues de 1.40, 1.21, 1.38, 1.27 y 1.29. Es la estimación más honesta del error en producción para un modelo que predice viajes futuros.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El pliegue que tenga **JFK Airport** como zona de origen: sus viajes son los más distintos (distancias de 13 a 45 km, tarifa fija hacia Manhattan y taxímetro hacia Brooklyn y Queens), y si el modelo nunca vio viajes que salen de JFK no puede aprender su patrón. Así fue: el pliegue 5 (Brooklyn y JFK Airport) tuvo un error de **6.43**, contra 1.19 a 1.31 en los demás. El promedio por zona es **2.2996** con desviación **2.0648**: el modelo generaliza bien a zonas de Manhattan nuevas (se parecen entre sí), pero no a una zona con reglas propias.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El **porcentaje de viajes subestimados en más de 20%**, porque mide exactamente el evento que genera un reclamo. El MAE (1.2754) y el RMSE (1.9213) son simétricos: castigan igual sobreestimar que subestimar, y miden dólares, no proporciones (un error de 3 dólares es grave en un viaje de 8 y nada en uno de 70). En la prueba, **26 de 595 viajes (4.37%)** se subestiman en más de 20%: unos 44 reclamos por cada 1000 viajes. El RMSE es mayor que el MAE porque castiga más los errores grandes, lo que indica que hay algunos viajes con errores mucho mayores que el promedio.

**Lo que confirmó la tabla, esquema por esquema:**

| esquema | MAE promedio | desviación | pliegues | peor pliegue |
|---|---|---|---|---|
| una sola división (Semana 10), gradient boosting | 1.2754 | — | — | — |
| al azar, gradient boosting | 1.2288 | 0.0361 | 1.28, 1.24, 1.16, 1.24, 1.23 | 1.28 |
| al azar, bosque | 1.2880 | 0.0291 | 1.30, 1.27, 1.25, 1.34, 1.28 | 1.34 |
| temporal, gradient boosting | 1.3099 | 0.0693 | 1.40, 1.21, 1.38, 1.27, 1.29 | 1.40 |
| por zona, gradient boosting | 2.2996 | 2.0648 | 1.27, 1.31, 1.29, 1.19, 6.43 | 6.43 (Brooklyn y JFK) |

Métricas en la prueba de la Semana 10: MAE 1.2754, RMSE 1.9213, subestimados en más de 20%: 4.37%.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy de **0.9508** (95.1%, la proporción de pagos legítimos: 295 fraudes de 6000 son el 4.9%) y recall de **0** (no detecta ningún fraude; tampoco tiene precisión, porque nunca predice fraude).

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy pasa de 0.9508 a **0.9712**: apenas **2 puntos**. El recall pasa de 0 a **0.4610**: detecta el **46%** de los fraudes (la logística, 26%), con una precisión de 0.9051. El recall describe mucho mejor lo que gana la empresa: casi la mitad de los fraudes, que antes pasaban todos, ahora se detectan. La accuracy apenas se mueve porque el 95% de los pagos ya estaban "bien" con la línea base.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con el umbral 0.5 deja pasar **159 de 295 fraudes** (detecta 136) con solo 14 falsas alarmas: un costo de 159 × 40 + 14 × 3 = **6402 dólares**. El paso 5 eligió el umbral **0.1**: detecta **173 fraudes**, deja pasar **122** y genera **222 falsas alarmas**, con un costo de **5546 dólares**, el menor de la lista. Como un fraude no detectado cuesta 13 veces más que una falsa alarma, conviene bajar el umbral y aceptar más falsas alarmas a cambio de detectar más fraudes.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

**Sube**, porque las falsas alarmas se vuelven caras y conviene marcar menos pagos. Con 20 dólares por falsa alarma, los costos son: 0.05 → 15 760, 0.1 → 9320, 0.15 → 7820, 0.2 → 7300, 0.3 → 6780 y **0.5 → 6640**: el umbral elegido pasa de 0.1 a **0.5**. El mejor umbral no es una propiedad del modelo, sino de los costos del negocio.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

_(tu respuesta)_

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

_(tu respuesta)_

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

_(tu respuesta)_

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La **tarifa** casi se podría calcular por adelantado: si las features son zona de origen, zona de destino, hora, día y pocas más, el número de combinaciones es finito (8 × 8 zonas × 24 horas × 2 tipos de día, por ejemplo), y se puede precalcular una tabla cada noche y consultarla al pedir el viaje. La distancia exacta complica eso, así que en la práctica también se sirve en línea, aunque es barata. El de **fraude** tiene que responder **en el momento**: depende de datos del pago que solo existen cuando ocurre (monto, tarjeta nueva, intentos fallidos recientes), y la decisión de bloquear o aprobar tiene que tomarse antes de cerrar el cobro, en un tiempo del orden de cientos de milisegundos para no demorar al pasajero. Una evaluación nocturna de fraude solo serviría para revisar después, cuando el dinero ya se fue.
