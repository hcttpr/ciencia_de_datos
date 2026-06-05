import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(123)

n = 300

datos_red = pd.DataFrame({
    "duracion_ms": np.random.normal(50, 10, n),
    "paquetes_enviados": np.random.normal(100, 20, n),
    "errores_checksum": np.random.poisson(2, n),
    "latencia_avg": np.random.normal(15, 5, n),
    "jitter": np.random.normal(2, 0.5, n),
    "uso_memoria_sw": np.random.normal(40, 10, n),
    "peticiones_http": np.random.normal(200, 50, n)
})

datos_red["bytes_enviados"] = (
    datos_red["paquetes_enviados"] * 1500
    + np.random.normal(0, 500, n)
)

datos_red["reintentos_tcp"] = (
    datos_red["errores_checksum"] * 1.5
    + np.random.normal(0, 0.5, n)
)

datos_red["carga_cpu_router"] = (
    datos_red["paquetes_enviados"] * 0.4
    + datos_red["latencia_avg"] * 0.2
)

scaler = StandardScaler()
pca_red = PCA()

X_scaled = scaler.fit_transform(datos_red)
componentes = pca_red.fit_transform(X_scaled)

varianza = pca_red.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(varianza)+1),
    varianza,
    marker="o"
)

plt.xlabel("Componentes Principales")
plt.ylabel("Varianza Explicada")
plt.title("Scree Plot - Tráfico de Red")

plt.grid(True)

plt.show()

plt.figure(figsize=(12,8))

plt.scatter(
    componentes[:,0],
    componentes[:,1],
    alpha=0.5
)

for i, variable in enumerate(datos_red.columns):
    plt.arrow(
        0, 0,
        pca_red.components_[0, i] * 4,
        pca_red.components_[1, i] * 4,
        head_width=0.05
    )

    plt.text(
        pca_red.components_[0, i] * 4.2,
        pca_red.components_[1, i] * 4.2,
        variable
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot PCA - Tráfico de Red")

plt.grid(True)

plt.show()