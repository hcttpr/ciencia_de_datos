import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n = 500

temperatura = np.random.normal(100, 10, n)

sensores = pd.DataFrame({
    "sensor1_temp": temperatura,
    "sensor2_temp": temperatura + np.random.normal(0, 2, n),
    "sensor3_presion": temperatura * 0.5 + np.random.normal(0, 5, n),
    "sensor4_presion": temperatura * 0.45 + np.random.normal(0, 5, n),
    "sensor5_vibracion": np.random.normal(20, 3, n),
    "sensor6_vibracion": np.random.normal(21, 3, n),
    "sensor7_flujo": np.random.normal(50, 8, n),
    "sensor8_flujo": np.random.normal(49, 8, n),
    "sensor9_CO2": temperatura * 0.3 + np.random.normal(0, 2, n),
    "sensor10_CO2": temperatura * 0.35 + np.random.normal(0, 2, n),
    "sensor11_humedad": np.random.normal(60, 7, n),
    "sensor12_humedad": np.random.normal(62, 7, n)
})


corr = sensores.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlación entre sensores")


scaler = StandardScaler()
pca = PCA()

X_scaled = scaler.fit_transform(sensores)
componentes = pca.fit_transform(X_scaled)

varianza = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(varianza)+1),
    varianza,
    marker="o"
)

plt.xlabel("Componentes")
plt.ylabel("Varianza explicada")
plt.title("Scree Plot")

plt.show()

varianza_acumulada = np.cumsum(varianza)

n_componentes = np.argmax(
    varianza_acumulada >= 0.85
) + 1

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(sensores.columns))],
    index=sensores.columns
)

plt.figure(figsize=(10,7))

plt.scatter(
    componentes[:,0],
    componentes[:,1],
    alpha=0.5
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot PCA (Observaciones)")

plt.show()