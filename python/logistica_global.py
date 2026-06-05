import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

np.random.seed(123)

n = 600

df = pd.DataFrame({
    "distancia_km": np.random.normal(500, 150, n),
    "tipo_transporte": np.random.choice(
        ["Terrestre", "Aereo"],
        size=n
    ),
    "clima": np.random.choice(
        ["Normal", "Lluvia", "Tormenta"],
        size=n
    )
})

df["tiempo_entrega_hrs"] = np.where(
    df["tipo_transporte"] == "Terrestre",
    df["distancia_km"] / 60 + np.random.normal(5, 2, n),
    df["distancia_km"] / 300 + np.random.normal(2, 1, n)
)

# Costos
df["costo"] = np.where(
    df["tipo_transporte"] == "Terrestre",
    df["distancia_km"] * 1.5,
    df["distancia_km"] * 3.5
)

indices = np.random.choice(df.index, 20, replace=False)
df.loc[indices, "distancia_km"] = np.nan

mediana = df["distancia_km"].median()

df["distancia_km"] = df["distancia_km"].fillna(mediana)

media = df["tiempo_entrega_hrs"].mean()
desviacion = df["tiempo_entrega_hrs"].std()

correlacion = df["distancia_km"].corr(
    df["tiempo_entrega_hrs"],
    method="pearson"
)

resumen = df.groupby("tipo_transporte").agg({
    "tiempo_entrega_hrs": "mean",
    "costo": "mean"
})

terrestre = df[df["tipo_transporte"] == "Terrestre"]["tiempo_entrega_hrs"]

aereo = df[df["tipo_transporte"] == "Aereo"]["tiempo_entrega_hrs"]

t_stat, p_value = ttest_ind(
    terrestre,
    aereo
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="distancia_km",
    y="tiempo_entrega_hrs",
    hue="tipo_transporte"
)

plt.title("Distancia vs Tiempo de Entrega")
plt.xlabel("Distancia (km)")
plt.ylabel("Tiempo de Entrega (hrs)")

plt.show()
    
plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="tipo_transporte",
    y="tiempo_entrega_hrs"
)

plt.title("Comparación de Tiempos de Entrega")

plt.show()