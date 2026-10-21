# Ejercicio 3: EDA del proyecto (fallas de SSD)

## Datos

El pipeline del proyecto deja la tabla para modelar (`OBT_MC1`) en Snowflake, y en este momento no hay acceso a esa cuenta. Por eso se usan los datos públicos de Alibaba, del mismo problema: <https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data>.

| Archivo | Contenido |
| --- | --- |
| `ssd_failure_tag.csv` | 18.387 SSD que **fallaron** (todas las filas tienen `failure = 1`), con atributos SMART |
| `location_info_of_ssd.csv` | Los 965.495 SSD, con `app`, `model`, `rack_id`, `node_id`, `slot_id` |

Como `ssd_failure_tag` no trae discos sanos, el script cruza los dos archivos por `model` y `disk_id`. Cada disco que no aparece en la lista de fallas queda con `failure = 0`. Todas las fallas encuentran su disco.

**Objetivo:** `failure` (1 = el SSD falló, 0 = no).

## Cómo correrlo

Desde `semana-08-sesion-02/`:

```
docker compose up --build eda
```

Los gráficos quedan en `data-contenedor/eda/`.

## Hallazgos

1. **Clases desbalanceadas.** Hay 18.387 fallas contra 947.108 discos sanos, una tasa de falla del 1,90 %. La regla del IQR marca todas las fallas como atípicas, porque Q1 = Q3 = 0, así que no sirve con un objetivo binario. Para el modelo: dividir los datos manteniendo la proporción de clases y medir con recall o PR-AUC, no con accuracy.
2. **El modelo de SSD y la aplicación cambian mucho el riesgo.** C1 falla un 5,2 % y A1 un 0,26 %, unas 20 veces menos (`tasa_falla_por_modelo.png`). Por aplicación, DAE falla un 7,6 % y SS un 0,6 %. Para el modelo: codificar `model` y `app` como categóricas (one-hot).
3. **`slot_id` es ruidosa.** El 25 % está vacío, y esos discos fallan menos (1,2 % contra 2,1 %). Además, 14 slots tienen menos de 100 discos, con tasas engañosas como el 100 % de un slot que tiene un solo disco. Para el modelo: agregar una columna que indique si falta el slot y agrupar los slots raros.

**Limitación:** las 31 columnas SMART solo existen para los discos que fallaron, así que no se pueden usar como variables con estos archivos. El modelo aprendería a reconocer "tiene datos SMART" en vez de "va a fallar".
