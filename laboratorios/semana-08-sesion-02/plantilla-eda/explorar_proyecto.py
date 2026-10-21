import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# Datos publicos de Alibaba (github.com/alibaba-edu/dcbrain, ssd_open_data).
# ssd_failure_tag solo trae SSD que fallaron, asi que se cruza con
# location_info (todos los SSD) para marcar cada disco como fallado o sano.
ubicacion = pd.read_csv("location_info_of_ssd.csv/location_info_of_ssd.csv")
fallas = pd.read_csv("ssd_failure_tag.csv/ssd_failure_tag.csv")

datos = ubicacion.merge(
    fallas[["model", "disk_id", "failure"]], on=["model", "disk_id"], how="left"
)
datos["failure"] = datos["failure"].fillna(0).astype(int)

# Variable que el modelo va a predecir: 1 si el SSD fallo, 0 si no.
OBJETIVO = "failure"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(datos[OBJETIVO], bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: tasa de falla por modelo de SSD (variable categorica).
por_modelo = datos.groupby("model")[OBJETIVO].mean().sort_values()

plt.figure()
plt.bar(por_modelo.index, por_modelo.values * 100)
plt.title("Tasa de falla por modelo de SSD")
plt.xlabel("Modelo")
plt.ylabel("% de discos que fallaron")
plt.savefig("figuras/tasa_falla_por_modelo.png")
plt.close()

# Paso 5: consultas que confirman los hallazgos.

# Hallazgo 1: clases desbalanceadas.
print("\nDiscos por clase")
print(datos[OBJETIVO].value_counts())
print(f"Tasa de falla: {datos[OBJETIVO].mean():.2%}")

# Hallazgo 2: el modelo y la app cambian mucho el riesgo.
print("\nFalla por modelo")
print(datos.groupby("model")[OBJETIVO].agg(["mean", "sum", "count"]).sort_values("mean"))
print("\nFalla por app")
print(datos.groupby("app")[OBJETIVO].agg(["mean", "sum", "count"]).sort_values("mean"))

# Hallazgo 3: slot_id tiene faltantes y slots con muy pocos discos.
print("\nFalla con slot_id vacio vs. lleno")
print(datos.groupby(datos["slot_id"].isna())[OBJETIVO].agg(["mean", "count"]))
por_slot = datos.groupby("slot_id")[OBJETIVO].agg(["mean", "count"])
print("\nSlots con menos de 100 discos")
print(por_slot.query("count < 100"))

# Los atributos SMART solo existen para los discos que fallaron.
smart = fallas.filter(regex="^[rn]_").columns
print("\nColumnas SMART en ssd_failure_tag:", len(smart))
print("Discos sanos con datos SMART: 0 (el archivo solo trae fallas)")
