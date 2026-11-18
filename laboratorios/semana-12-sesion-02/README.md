# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

1. **Guardar el modelo entrenado** (serializarlo, por ejemplo con `pickle` o `joblib`), junto con su preparación, para no tener que reentrenarlo en cada uso.
2. **Envolverlo en un servicio web**: una API (Flask o FastAPI) que cargue el modelo una sola vez al arrancar y exponga una ruta que reciba los datos de entrada como JSON y devuelva la predicción.
3. **Definir el contrato de entrada y salida**: qué campos recibe, de qué tipo, y qué responde; validar la entrada antes de llegar al modelo.
4. **Empaquetarlo en un contenedor** (Docker) con sus dependencias en versiones fijas.
5. **Desplegarlo** en un servidor o en la nube, donde quede encendido y accesible para otras aplicaciones, y monitorearlo.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que el servicio corra **igual en cualquier máquina**, no solo en la de quien lo escribió. "En mi máquina funciona" depende de la versión de Python, de las librerías (un modelo guardado con joblib solo carga bien con la misma versión de scikit-learn), del sistema operativo y de archivos que están en la laptop del autor. La imagen empaqueta el código, el modelo, el intérprete y las dependencias exactas, así que se puede llevar tal cual a un servidor o a la nube, reproducirla, escalarla (correr varias copias) y reemplazarla por una versión nueva sin instalar nada a mano.

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un **servicio de predicción en línea**: una API siempre encendida, con el modelo cargado en memoria, que exponga una ruta como `POST /predecir`. La app le mandaría, en el momento en que el pasajero elige el destino, los datos del viaje que conoce en ese instante como JSON: fecha y hora, zona de origen, zona de destino, distancia estimada, pasajeros y método de pago. El servicio calcularía las features derivadas (hora como ciclo, hora pico, fin de semana, tarifa fija) con la misma función que se usó para entrenar, y respondería con la tarifa estimada en milisegundos.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo "pasa" pero el servicio **no hace nada**: `/predecir` responde siempre `{"tarifa_estimada": 0.0}` sin tocar el modelo, y el esquema `Viaje` solo declara las zonas, así que ignora la fecha, la distancia y los pasajeros. `/salud` dice que el modelo está cargado, pero eso solo significa que el archivo existe, no que se use. Y los 3 `passed` engañan: dos de las pruebas no tienen ningún `assert` todavía, así que pasan siempre. Un código 200 y pruebas verdes no garantizan que la respuesta sea correcta.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

- **Sin `distancia_km`**: lo acepta y estima una tarifa (200), porque en el esquema es opcional (`float | None = None`) y la preparación del pipeline la imputa con la mediana de entrenamiento. Probado: un viaje de Midtown a Harlem sin distancia recibió `{"tarifa_estimada": 24.26}`. (Es una tarifa razonable "en promedio", no para ese viaje en particular.)
- **Sin `zona_origen`**: lo rechaza con **422 Unprocessable Entity**, porque la zona es obligatoria en el esquema; la respuesta dice qué campo falta (`"loc": ["body", "zona_origen"], "msg": "Field required"`).

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo se equivocaría **sin dar ningún error**. Aprendió que `es_hora_pico = 1` significa "7 a 9 y 16 a 19 en día de semana" y que esos viajes cuestan más por kilómetro. Si el servicio usara otras horas (por ejemplo, 17 a 20), un viaje de las 20 h recibiría el ajuste de hora pico que no le corresponde, y uno de las 7 no; las predicciones se desviarían de forma sistemática, y nadie lo notaría porque el modelo sigue devolviendo números con forma razonable. Es el *training-serving skew*: por eso el servicio importa `agregar_features` de `features_taxis.py`, la misma función con que se construyeron las features, en lugar de reescribirla.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce **FastAPI**, no mi código: al recibir la petición, valida el JSON contra el esquema `Viaje` (con pydantic) y, como falta un campo obligatorio, responde 422 con el detalle del error **antes de llamar a la función `predecir`**. La petición no llega al modelo. Por eso la prueba funciona con cualquier modelo, y por eso declarar bien el esquema protege al modelo de entradas incompletas o mal escritas (una fecha como `"5 de noviembre"` también recibe 422).

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque sin `--host`, uvicorn escucha en `127.0.0.1`, el *localhost* **del contenedor**. Las peticiones que llegan desde fuera (desde la máquina anfitriona, a través del `-p 8000:8000`) entran por la interfaz de red del contenedor, no por su localhost, así que nadie las atiende y `curl` falla aunque el log diga que uvicorn está corriendo. Con `--host 0.0.0.0` escucha en todas las interfaces y el puerto publicado funciona. (Respondido a partir de la configuración: no construí la imagen con Docker en esta corrida; el `CMD` ya quedó con `"--host", "0.0.0.0"`.)

**Lo que respondió el servicio:**

- `/salud`: `{"estado":"ok","modelo_cargado":true}`.
- Viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta): `{"tarifa_estimada":70.39}` (la tarifa fija es 70.00; la red se acerca).
- Viaje sin zona de origen: código **422** y
  ```json
  {"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}
  ```

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

- **Pago riesgoso** (2026-11-04 02:40, app, 41.30 dólares, 8.6 km, tarjeta nueva, 2 intentos fallidos, 0 viajes previos): `{"probabilidad_fraude": 0.9996, "revisar": true}`.
- **Pago tranquilo** (2026-11-04 14:10, tarjeta, 19.80 dólares, 6.3 km, tarjeta conocida, 0 intentos fallidos, 22 viajes previos): `{"probabilidad_fraude": 0.0159, "revisar": false}`.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no tiene que conocer el umbral ni la regla de decisión: recibe directamente qué hacer (mandar a revisión o cobrar), y todas las apps que llamen al servicio aplican la misma regla. La decisión de negocio queda en un solo lugar, junto al modelo. Si mañana el umbral pasa a 0.20, solo cambia la constante `UMBRAL` en `servicio_fraude.py` (o, mejor, una variable de entorno o configuración): se reconstruye y se vuelve a desplegar el servicio, sin tocar la app. Conviene además registrar qué umbral se usó (por ejemplo, devolverlo en la respuesta o en los logs) para saber después con qué regla se tomó cada decisión.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí. El pago 70001 del lote de ayer tuvo probabilidad **0.2628** en `pagos_puntuados.csv`, y mandado al servicio dio exactamente **0.2628**. Para que coincida tiene que ser igual: el **modelo** (el mismo archivo: los dos `modelo_fraude.joblib` tienen la huella `377bc8b74cf7`, porque `entrenar_fraude.py` es el mismo con los mismos datos y semilla), la **preparación de las features** (los dos importan `agregar_features` y `columnas` de `preparacion_pagos.py`, idénticos), los **datos de entrada** del pago, y la **versión de scikit-learn** con que se carga el modelo.

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde **422** con `{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}`. Lo produce la **línea 34** de `servicio_fraude.py`, `monto: float`, en el esquema `Pago`: FastAPI valida la entrada con ese tipo y rechaza la petición antes de entrar a `evaluar_pago`. No hay ninguna línea de código que escriba esa respuesta: sale del esquema.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

_(tu respuesta)_

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Habría que **versionar el modelo y registrar la versión en cada predicción**. Para saber qué modelo produjo una predicción, el servicio debería devolver (y guardar en sus logs) un identificador del modelo, por ejemplo la huella del archivo o el `run_id` y la versión en el registro de modelos de MLflow, junto con la predicción, la entrada y la hora. Para volver al anterior, el archivo no debería reemplazarse sin más: cada versión tiene que quedar guardada en un registro de modelos (MLflow Model Registry), con su `run_id`, sus métricas y una etiqueta o alias de cuál está en producción. Volver atrás es mover ese alias a la versión anterior y reiniciar el servicio (o tener una imagen de contenedor por versión del modelo y desplegar la anterior). Eso es parte del gobierno de modelos: quién aprobó cada versión, cuándo se puso en producción y por qué.
