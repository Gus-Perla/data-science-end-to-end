# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

- **Batch (por lotes):** las predicciones se calculan **por adelantado**, periódicamente (cada hora, cada noche), para muchos casos a la vez, y se guardan en una tabla o base de datos; cuando alguien necesita una predicción, la lee de ahí. Las features vienen de datos históricos ya procesados. Ejemplo: las recomendaciones de películas o productos que se recalculan cada noche para cada usuario, o el puntaje de riesgo de abandono de cada cliente que se envía al equipo comercial cada lunes.
- **Online (en línea, en tiempo real):** la predicción se calcula **en el momento en que se pide**, normalmente a través de una API que recibe los datos del caso y responde en milisegundos, con features que pueden incluir información de ese instante. Ejemplo: decidir si una transacción con tarjeta es fraude mientras se procesa el pago, o estimar el tiempo de llegada en una app de transporte.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

El batch solo puede predecir casos **que se conocen de antemano**, con datos **de hasta la última corrida**: no responde a un caso nuevo (un usuario que acaba de registrarse, un destino que nadie reservó) ni a lo que acaba de pasar (lo que el usuario hizo hace un minuto). Sus predicciones pueden quedar desactualizadas. El online lo resuelve calculando cada predicción con los datos del momento. Lo que cuesta es **infraestructura y operación**: un servicio que esté siempre encendido y responda en milisegundos, que escale con el tráfico, que tenga monitoreo y alertas si se cae, y features disponibles en tiempo real (a veces hace falta un *feature store* o streaming). También es más difícil de probar y de depurar, y cada predicción se paga en el momento en lugar de hacerse en bloque cuando la máquina está libre.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

La **tarifa** se puede calcular por adelantado en parte: los viajes reservados para el día siguiente se conocen la noche anterior, y se pueden estimar todos en un lote (el Ejercicio 1). Pero un viaje que se pide en el momento, con un destino cualquiera, no se conoce de antemano: ese necesita una estimación en línea, en menos de un segundo (lo que el pasajero tolera antes de que la app parezca lenta, idealmente unos cientos de milisegundos). El de **fraude**, si la decisión es bloquear el pago, tiene que responder en el momento en que se cobra, en milisegundos, porque depende de datos que solo existen entonces (monto, intentos fallidos, tarjeta nueva). Si la decisión es solo revisar después, se puede hacer por lotes cada madrugada (el Ejercicio 2), aceptando que el dinero ya se cobró.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque esas dos pruebas no comprobaban nada todavía: `test_una_prediccion_por_viaje` y `test_viaje_incompleto_tambien_se_estima` solo llaman a `predecir_lote` y tienen un comentario `TODO` en lugar de un `assert`. Una prueba sin `assert` pasa siempre que el código no lance una excepción. La única que comprueba algo, `test_columnas_de_salida`, falla porque la tabla solo tiene `id_viaje`. Que una prueba pase no dice nada si no tiene una afirmación que pueda fallar.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Que les estime una tarifa, como a los demás. El script le pasa los viajes al **pipeline completo** que se guardó, y ese pipeline incluye la preparación de la Semana 9: el `SimpleImputer` llena `pasajeros` vacío con la mediana de entrenamiento y `metodo_pago` vacío con el valor más frecuente, aprendidos al entrenar. Así fue: los 60 viajes tienen tarifa; los 4 sin pasajeros recibieron tarifas normales para su distancia (de 17.58 a 33.94) y los 2 sin método de pago, 29.59 y 60.79. Un lote nocturno no debería caerse ni saltarse viajes por un dato faltante: por la mañana nadie está mirando, y esos viajes quedarían sin tarifa.

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para la **trazabilidad**: si por la mañana un pasajero reclama una tarifa, o si se descubre que un modelo tenía un error, hay que saber qué modelo produjo cada estimación. La tabla se lee después de que el script terminó, y el modelo puede haberse reentrenado entre una noche y otra; sin la versión en cada fila no hay forma de saber de cuál salió el número ni de reproducirlo. También permite comparar estimaciones de distintas versiones y relacionarlas con el registro de MLflow. Es la huella MD5 del archivo del modelo: cambia si cambia un solo byte.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

- La del **TODO 4** fallaría si `predecir_lote` cambiara la cantidad o el orden de las filas: por ejemplo, si se agregara un `dropna()` que quite los viajes con datos faltantes, si se ordenara la salida por tarifa, o si `id_viaje` se tomara de otro lado (perdiendo la correspondencia con cada viaje).
- La del **TODO 5** fallaría si la preparación dejara de imputar (por ejemplo, si se predijera con un modelo sin el `SimpleImputer`, lo que haría fallar o dejar `NaN` en el viaje sin pasajeros), o si la columna `version_modelo` se llenara con otro valor (por ejemplo, una versión fija escrita a mano en lugar de la que se pasa como argumento).

**Lo que mostró el lote:**

- **Versión del modelo:** `826bc5c1aa22` (60 viajes leídos, 60 filas en `tarifas_estimadas.csv`).
- **Tarifa más baja:** **4.67**, viaje 90014, de Upper East Side a Upper East Side, 0.4 km.
- **Tarifa más alta:** **70.26**, viaje 90051, de Upper East Side a JFK Airport (tarifa fija); los otros dos viajes de tarifa fija dieron 69.94 y 69.82. La red no sabe que la tarifa fija es exactamente 70, solo se acerca.
- **Viajes sin pasajeros:** 90023 (Queens → Upper West Side, 10.58 km): 33.94; 90024 (Brooklyn → Harlem, 10.17 km): 29.88; 90025 (Midtown → Harlem, 8.40 km): 25.80; 90048 (Harlem → Upper East Side, 4.75 km): 17.58. Tarifas normales para su distancia, gracias a la imputación.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

**26 pagos** quedaron en la cola con el umbral de 0.10. Con la proporción de la Semana 11 (295 fraudes de 6000, 4.9%), en 300 pagos esperaría unos **15 fraudes** (14.75). Que la cola tenga más pagos que los fraudes esperados es coherente con el umbral bajo: en la Semana 11 ese umbral detectaba 173 fraudes con 222 falsas alarmas, así que una parte de la cola son pagos legítimos que se revisan por precaución.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de **detener el fraude antes de que ocurra**: cuando el equipo revisa la cola, el pago ya se cobró, el viaje ya se hizo y, si era una tarjeta robada, el contracargo llega igual. Además, el mismo estafador puede seguir usando la tarjeta durante todo el día hasta que se revise. La revisión solo sirve para recuperar o bloquear después. Para que eso importe, la revisión tendría que poder actuar **antes de completar el pago**: bloquear o pedir verificación (por ejemplo, un código o un método de pago alternativo) en el momento de cobrar. Ahí el modelo tiene que ser online y responder en milisegundos.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Prueba **solo la lógica de la cola** (el filtro y el orden), aislada del modelo. Con probabilidades escritas a mano se sabe exactamente cuál debe ser el resultado (4, 2, 3), y se pueden poner a propósito los casos difíciles, como un valor justo en el límite (0.10 entra) y uno por debajo (0.05 no). Si usara el modelo, el resultado dependería de los datos y del entrenamiento: cambiaría al reentrenar, la prueba no sabría qué esperar, sería lenta, y en GitHub Actions no existe el archivo `.joblib`. Además, si la prueba falla se sabe que el error está en `a_revisar` y no en el modelo.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con el umbral en 0.50 quedan **13 pagos** en la cola (la mitad que con 0.10). Las pruebas **siguen pasando todas**, porque ninguna depende de la constante `UMBRAL`: `test_cola_de_revision` llama a `a_revisar` con su propio umbral (0.10) y sus propias probabilidades, y las demás revisan la forma de la salida de `puntuar_lote`. Las pruebas protegen que el código haga lo que dice (filtrar con el umbral que se le pasa), no que el umbral sea el correcto; esa es una decisión de negocio que se toma con los costos de la Semana 11. (Después de probarlo dejé `UMBRAL` en 0.10.)

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

_(tu respuesta)_

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Tendría que existir un **servicio de predicción en línea**: un programa siempre encendido (por ejemplo, una API web con FastAPI, empaquetada en un contenedor) que cargue el modelo una vez al arrancar y exponga una dirección (un *endpoint*, como `POST /predecir`). La app le mandaría, en el momento de pedir el viaje, los datos del viaje en un JSON: zona de origen, zona de destino, distancia estimada, pasajeros, método de pago y la hora (o las features ya calculadas: `hora_sin`, `hora_cos`, `es_hora_pico`, `es_fin_de_semana`, `es_tarifa_fija`). El servicio validaría esos datos, correría el pipeline y respondería con la tarifa estimada y la versión del modelo, en milisegundos.
