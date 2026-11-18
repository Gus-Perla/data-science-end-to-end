# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

Es el proceso de punta a punta con el que una organización establece, implementa y mantiene **controles** sobre cómo se desarrollan, despliegan y usan sus modelos: desde la documentación y el control de versiones hasta las pruebas retrospectivas (*back-testing*) y el monitoreo. El artículo la divide en cinco componentes:

1. **Desarrollo y documentación del modelo**: cómo se construyó, con qué datos y supuestos.
2. **Inventario de modelos**: un registro central de todos los modelos en uso.
3. **Validación y *back-testing***: comprobar, antes y después de desplegar, que el modelo hace lo que debe.
4. **Despliegue y monitoreo**: cómo se pone en producción y cómo se vigila su desempeño.
5. **Roles, responsabilidades y flujos de trabajo**: quién desarrolla, quién valida, quién aprueba y en qué orden.

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Para cada modelo: su **propósito** (para qué se usa y qué decisión apoya), su **responsable** (quién es el dueño y a quién preguntar), su **metodología** (cómo funciona: algoritmo, datos, features, supuestos) y su **estado en el ciclo de vida** (en desarrollo, en validación, en producción, retirado). En la práctica conviene sumar la versión, las métricas con que se aprobó y la fecha. Alguien que no lo entrenó (un auditor, un regulador, otro equipo, el que llega a mantenerlo) los necesita para saber **qué modelos existen y cuáles están en uso**, evaluar su riesgo, entender por qué decide lo que decide, saber a quién acudir si falla y no usar por error uno retirado o no aprobado. Según el artículo, un buen inventario permite evaluar mejor el riesgo y tomar decisiones más rápidas e informadas sobre el uso de los modelos.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Con un archivo que se reemplaza, **no lo sabría**: el archivo de hoy ya no existe, y si el servicio no devolvía ninguna versión en la respuesta no hay forma de relacionar esa predicción con un modelo. Tampoco podría volver al anterior, porque lo sobrescribí. Hace falta un **registro de modelos**: cada entrenamiento se guarda como una versión numerada que nunca se borra, ligada a su corrida (datos, parámetros, métricas); un alias (`produccion`) indica cuál se sirve; y el servicio responde en cada predicción qué versión la calculó. Para saber qué modelo produjo la predicción de hoy, miro la `version_modelo` de esa respuesta (o del log). Para volver al anterior, muevo el alias a la versión anterior y reinicio el servicio.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

El error es `mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Registered model alias produccion not found.` Dice que **el registro no tiene ninguna versión** con el alias `produccion` (de hecho, no tiene ninguna versión): `registrar_modelo.py` entrenó y midió el modelo (MAE 1.1695) y guardó la corrida, pero sin el `TODO 1` no guardó el modelo ni lo registró, por eso dice `Version None` y `versiones.py` no lista nada. Y sin el `TODO 2`, `promover.py` no pone ningún alias, así que al buscarlo no lo encuentra.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Esperaba **dos versiones**, y la primera **sigue intacta**: registrar una versión nueva no reemplaza ni borra la anterior, solo agrega otra con el número siguiente. Así fue:

```
Version 1: red de 64 neuronas, MAE 1.1695, alias: -
Version 2: red de 8 neuronas, MAE 1.1877, alias: -
```

Ninguna tiene alias todavía: registrar no es lo mismo que poner en producción.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

- Con `produccion` en la versión 2, `/salud` dice `"version_modelo": "2"`.
- Al mover el alias a la versión 1 **sin reiniciar**, el servicio **sigue respondiendo la versión 2**: el modelo se carga una sola vez, al arrancar, y el servicio no vuelve a mirar el registro.
- **Después de reiniciarlo**, responde la versión 1.

Mover el alias es instantáneo, pero el cambio llega a producción recién cuando el servicio se reinicia (o se despliega una imagen nueva). Por eso conviene que el servicio diga qué versión sirve: el alias dice cuál *debería* estar, `/salud` dice cuál está.

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Antes del `TODO 4` falla **`test_cada_prediccion_dice_que_version_la_calculo`**: la respuesta era `{"tarifa_estimada": 20.0}` sin `version_modelo`. Comprueba que **cada predicción** lleva la versión que la calculó. `/salud` solo dice qué versión está cargada **ahora**; si la versión cambia (al reiniciar con el alias movido), una tarifa guardada por la app ayer no se puede relacionar con el `/salud` de hoy. La versión tiene que viajar con cada respuesta para que quien la reciba (o la guarde) pueda rastrearla después.

**Lo que respondió el servicio:**

Viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta):

| alias `produccion` en | servicio | `/salud` | `/predecir` |
|---|---|---|---|
| versión 2 (8 neuronas) | recién levantado | `"version_modelo": "2"` | `{"tarifa_estimada": 69.98, "version_modelo": "2"}` |
| versión 1 (64 neuronas) | sin reiniciar | `"version_modelo": "2"` | `{"tarifa_estimada": 69.98, "version_modelo": "2"}` |
| versión 1 (64 neuronas) | reiniciado | `"version_modelo": "1"` | `{"tarifa_estimada": 70.07, "version_modelo": "1"}` |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

La **versión 2, el gradient boosting** (recall 0.661, precisión 0.429). La versión 1 (logística, recall 0.644) entró primero porque no había ninguna en producción; la 2 la superó en recall y la reemplazó. La última registrada, la 3 (logística muy regularizada), tiene recall **0.102**: detecta muchos menos fraudes que la 2, así que la regla la rechazó. La regla no promueve "la más reciente", sino la que mejora a la que está en producción.

```
logistica: version 1, recall 0.644, precision 0.304, promovida: True
gradient-boosting: version 2, recall 0.661, precision 0.429, promovida: True
logistica-muy-regularizada: version 3, recall 0.102, precision 0.667, promovida: False
En produccion: version 2
```

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla mira **solo el recall**, y la logística muy regularizada (`C=0.001`) casi no se atreve a marcar fraudes: los pocos que marca suelen serlo (precisión 0.667), pero deja pasar casi el 90% (recall 0.102 contra 0.661 de producción). Si las falsas alarmas cuestan más que los fraudes no detectados, la regla tendría que comparar una métrica que pese ese costo: por ejemplo, el **costo total esperado** (`fraudes no detectados × costo de fraude + falsas alarmas × costo de falsa alarma`, como en la Semana 11), promoviendo la de menor costo; o exigir una precisión mínima y, entre las que la cumplan, la de mayor recall; o el F-beta con beta < 1. Probablemente también habría que revisar el umbral de 0.10, que se eligió con los costos anteriores.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En las **etiquetas de la versión 3** en el registro: `validacion = rechazada` y `comparada_con = 2`. Dicen que fue evaluada, que no pasó y contra qué versión se comparó. El porqué numérico está en las **métricas de su corrida** (enlazada desde la versión): recall 0.102 contra 0.661 de la versión 2. Así la decisión queda documentada en el registro, no en la memoria de quien corrió el script.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Ahora hay **6 versiones**, y en producción **sigue la versión 2**. Es lo esperado: los tres candidatos se vuelven a entrenar con los mismos datos y semillas, así que tienen exactamente las mismas métricas (versiones 4, 5 y 6, con recall 0.644, 0.661 y 0.102). Como la regla pide un recall **estrictamente mayor**, ni siquiera la versión 5 (un gradient boosting idéntico a la 2) la reemplaza; las tres quedan con `validacion = rechazada` y `comparada_con = 2`. El registro no borra nada y no promueve un empate: evita mover producción sin una mejora real, aunque se acumulan versiones repetidas.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

_(tu respuesta)_

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se **degrada en silencio**: sigue respondiendo rápido, con código 200 y números con forma razonable, pero cada vez más alejados de la realidad. Pasa porque el mundo cambia después del entrenamiento: **data drift** (cambian las entradas: más viajes al aeropuerto, zonas nuevas, otra distribución de distancias), **concept drift** (cambia la relación entre entradas y tarifa: sube el precio por kilómetro, una tarifa fija nueva), o problemas en el **pipeline de datos** (una columna que llega vacía o con otro nombre, una feature calculada distinta). `/salud` y la latencia no lo detectan, porque miden si el servicio está vivo, no si acierta. Me enteraría **monitoreando el modelo**, no solo el servicio: guardando cada predicción con su versión y comparándola con la tarifa real cuando el viaje termina (MAE por día y por versión, con alertas), y mientras no llegan las etiquetas, vigilando la distribución de las entradas y de las predicciones contra las de entrenamiento (con pruebas como Kolmogorov-Smirnov o el PSI), y la proporción de datos faltantes o inválidos.
