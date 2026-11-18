# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tu notebook de exploración de la Semana 8, Sesión 2: las features de hoy salen de tus 3 hallazgos.

Los laboratorios de hoy siguen siendo notebooks: diseñar una feature es probar varias ideas y mirar cuál aporta. La próxima sesión, las features que se queden pasan a un script, que es lo que corre en producción.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio traduce algo que alguien del negocio ya sabe que mueve el resultado (una regla de precios, un comportamiento de los clientes, una restricción operativa) a una columna que el modelo puede usar. Por eso se puede explicar en una frase, se sabe por qué debería funcionar y se espera que siga funcionando con datos nuevos. Una feature que solo mejora la métrica es una combinación encontrada por prueba y error: sube el puntaje en la validación, pero nadie sabe explicar por qué, puede ser casualidad del conjunto de datos o, peor, una fuga del objetivo, y se rompe cuando cambian los datos. Además, la feature con sentido de negocio tiene que existir en el momento de predecir y poder calcularse en producción.

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

Un ejemplo típico es el de clientes y compras: el negocio sabe que un cliente que lleva mucho tiempo sin comprar está a punto de irse. Ese conocimiento no está en ninguna columna cruda (solo están las fechas de cada compra), pero se traduce a una columna de **recencia**: días desde la última compra hasta la fecha de corte (`fecha_corte - max(fecha_compra)` por cliente), junto con la frecuencia (número de compras en una ventana) y el monto. El modelo no tiene que "descubrir" la regla a partir de fechas sueltas: se la damos ya calculada, en las unidades en que el negocio piensa.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Construyendo columnas nuevas a partir de las que ya existen, usando lo que sabemos del negocio:

- **Tarifa fija:** una columna binaria `es_tarifa_fija` que vale 1 cuando el viaje va de JFK a una zona de Manhattan o al revés. Con ella, un modelo lineal puede aprender un ajuste propio para esos viajes en lugar de forzarlos a seguir la recta de la distancia.
- **Hora:** de `fecha_hora_inicio` (que el modelo no puede usar como texto) se sacan la hora, el día de la semana, `es_fin_de_semana` y `es_hora_pico` (horas pico en días de semana). Así el modelo puede aprender que en hora pico el kilómetro cuesta más (por el tráfico, que alarga la duración). Para la hora como número continuo se usa su seno y coseno, para que las 23 y las 0 queden juntas.

Todas existen en el momento en que se pide el taxi, así que se pueden usar en producción.

## Ejercicio 1: features de fecha y hora

Notebook: [`starter/ejercicio1_features_tiempo.ipynb`](./starter/ejercicio1_features_tiempo.ipynb), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. Ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como número, la hora 23 y la hora 0 están a **23 unidades**, la distancia máxima posible: el modelo las trataría como las horas más distintas del día. En el reloj están a **una hora**, igual que las 11 y las 12. Con `hora_sin` y `hora_cos` la hora es un punto en un círculo, y el Paso 5 lo confirma: 23 y 0 quedan a **0.26**, exactamente la misma distancia que 11 y 12 (0.26).

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

Más alta. La tarifa del taxímetro tiene un componente por distancia y otro por tiempo; en hora pico el tráfico hace que cada kilómetro tome más minutos, así que se paga más por kilómetro. La salida lo confirma: la mediana de la tarifa por km es **3.52 en hora pico contra 3.22 fuera de ella** (antes del TODO 3 había un solo grupo, con mediana 3.35). Los minutos por kilómetro explican la diferencia: 3.64 en hora pico contra 2.79 fuera.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La parte de la duración que se debe al **tráfico**: cuántos minutos por kilómetro va a tomar el viaje según el momento del día. La duración tiene dos componentes: la distancia (que ya está en `distancia_km`) y la velocidad. `es_hora_pico` le da al modelo una señal de la velocidad esperada (3.64 min/km en hora pico contra 2.79 fuera), sin usar información del futuro. Lo que no puede traer es lo imprevisible de cada viaje en particular: un accidente, una desviación o un semáforo.

**Lo que confirmó la salida, TODO por TODO:**

- **TODO 1 y 2:** el primer viaje (2026-09-06 13:46) queda con `hora = 13`, `dia_semana = 6` (domingo) y `es_fin_de_semana = 1`.
- **TODO 3:** 888 de 2975 viajes son de hora pico. Tarifa por km (mediana, sin tarifa fija ni distancia 0): **3.22 fuera de hora pico y 3.52 en hora pico**.
- **TODO 4 (Paso 5):** 23 y 0 están a 23 como número y a **0.26** como ciclo; 11 y 12 están a 1 como número y a **0.26** como ciclo. En el reloj ambos pares están igual de cerca.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Notebook: [`starter/ejercicio2_features_zona.ipynb`](./starter/ejercicio2_features_zona.ipynb), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. La última celda imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

**300 viajes**, y todos tienen tarifa mínima **70.00** y máxima **70.00**: la tarifa no varía nada dentro del grupo. Para un modelo lineal, esta columna es casi perfecta: le permite aprender que esos viajes tienen un precio fijo, que no depende de la distancia. Pero el modelo lineal solo suma un ajuste constante (`coeficiente · es_tarifa_fija`) encima de lo que ya predice con la distancia; como esos viajes miden entre 13 y 45 km, la parte de la distancia seguiría moviendo la predicción. Para capturarlo del todo hace falta una interacción (por ejemplo `distancia_km · (1 - es_tarifa_fija)`) o un modelo de árboles, que puede separar el grupo y predecir 70 directamente.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea **8 columnas** (una por zona: `origen_Brooklyn`, `origen_Harlem`, ..., `origen_Upper West Side`); la codificación ordinal crea **1 columna** con valores de 0 (Brooklyn) a 7 (Upper West Side). Para una regresión lineal usaría **one-hot**: cada zona recibe su propio coeficiente. La ordinal le da información falsa: le dice al modelo que las zonas tienen un orden y que la distancia entre ellas significa algo, por ejemplo que Upper West Side (7) es "siete veces Brooklyn" o que JFK (2) está "entre" Harlem (1) y Lower Manhattan (3), solo por el orden alfabético. El modelo tendría que asignar un efecto que crece de forma lineal con ese número, algo que no tiene sentido. (Un modelo de árboles sí tolera la ordinal, porque solo hace cortes.)

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

- `duracion_min`: se conoce cuando el viaje termina.
- `propina`: se decide después de conocer la tarifa.
- `metodo_pago`: en un taxi de calle se elige al pagar, al final (en una app podría conocerse antes, pero no lo asumo).
- `tarifa`: es el objetivo.

`distancia_km` real tampoco existe exactamente al pedir, pero se puede estimar con la ruta entre origen y destino, así que se usa la estimada. Si entrenara con duración o propina, el modelo se vería excelente en la evaluación (la duración tiene correlación 0.956 con la tarifa) porque estaría "viendo" parte de la respuesta, pero en producción esas columnas no existen: o no se puede correr, o hay que llenarlas con algo y las predicciones se desploman. Es fuga de información.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de tu exploración de la Semana 8 y copia la plantilla [`plantilla-features/features_proyecto.ipynb`](./plantilla-features/features_proyecto.ipynb) a `psets/proyecto/modelado/`, al lado de tu notebook de exploración. Todo lo que pide este ejercicio queda en el notebook: el código de cada feature, la evidencia y la justificación en las celdas de texto.

### Paso 1: el momento de la predicción (5 minutos)

En la primera celda de texto, escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

### Paso 2: construye las 2 features (15 minutos)

Crea 2 columnas nuevas, derivadas de las existentes, cada una en su celda y con su evidencia (una correlación, una diferencia de promedios). Al menos una tiene que salir de uno de tus hallazgos de la exploración. Si pruebas una idea que no aporta, deja la celda y escribe por qué la descartas: también es evidencia.

### Paso 3: justifica cada feature (5 minutos)

Debajo de cada feature: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta, y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar. Antes de hacer commit, reinicia el kernel y corre todo.

**Ruta de tu notebook de features en tu fork:**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los viajes de entrenamiento. Si calculo la media y la desviación con todos, la información de los viajes de prueba se filtra al preprocesamiento: el conjunto de prueba deja de ser "datos que el modelo nunca vio" y la evaluación queda un poco optimista. Con distancia el efecto suele ser pequeño, pero el principio es el mismo que con cualquier fuga, y en transformaciones más agresivas (imputar con la media, seleccionar features, *target encoding*) la diferencia puede ser grande. Lo correcto es ajustar el escalador con `fit` solo en entrenamiento y aplicar ese mismo escalador (`transform`) a prueba y a producción; un `Pipeline` de scikit-learn lo hace automáticamente, incluso dentro de la validación cruzada.
