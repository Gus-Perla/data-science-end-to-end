import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
errores = (viajes["tarifa"] == 999.99) | (viajes["duracion_min"] < 0)
limpios = viajes[~errores]
print(f"Filas originales: {len(viajes)}")
print(f"Errores confirmados quitados: {errores.sum()}")
print(f"Filas que quedan: {len(limpios)}")

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=5, alpha=0.5)
plt.xlabel("distancia_km")
plt.ylabel("tarifa")
plt.title("Distancia contra tarifa")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# La linea horizontal en 70 son viajes con tarifa fija (JFK).
tarifa_fija = limpios[limpios["tarifa"] == 70.0]
print(f"\nViajes con tarifa exactamente 70.0: {len(tarifa_fija)}")
print(tarifa_fija[["zona_origen", "zona_destino"]].value_counts())
print(tarifa_fija["distancia_km"].describe().round(2))

# Los JFK que no son tarifa fija van a Brooklyn o Queens, no a Manhattan.
jfk = (limpios["zona_origen"] == "JFK Airport") | (limpios["zona_destino"] == "JFK Airport")
print(f"Viajes de/hacia JFK sin tarifa 70: {(jfk & (limpios['tarifa'] != 70.0)).sum()}")

# La columna vertical en x = 0 son viajes con distancia 0 pero tarifa
# y duracion normales: falla del registro de distancia.
distancia_cero = limpios[limpios["distancia_km"] == 0]
print(f"\nViajes con distancia_km = 0: {len(distancia_cero)}")
print(distancia_cero[["duracion_min", "tarifa"]].describe().round(2))

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
propinas = limpios.groupby("metodo_pago").agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    propina_cero=("propina", lambda p: (p == 0).sum()),
)
print("\n", propinas.round(2))

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina"]
print("\nCorrelacion con la tarifa:")
print(limpios[columnas].corrwith(limpios["tarifa"]).round(3))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print(f"\nPasajeros vacio: {limpios['pasajeros'].isna().sum()}")
print(f"Pasajeros en 0: {(limpios['pasajeros'] == 0).sum()}")
