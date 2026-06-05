import pandas as pd
import numpy as np

# Reproducibilidad
np.random.seed(42)

n = 500

# Datos simulados
temperatura = np.random.normal(75, 5, n)
presion = np.random.normal(100, 10, n)

# Relación entre temperatura y errores
tasa_error = 2 + 0.15 * (temperatura - 75) + np.random.normal(0, 0.8, n)

turno = np.random.choice(
    ['Matutino', 'Vespertino'],
    size=n
)

# Crear DataFrame
df = pd.DataFrame({
    'temperatura': temperatura,
    'presion': presion,
    'tasa_error': tasa_error,
    'turno': turno
})

# Introducir algunos valores faltantes
df.loc[np.random.choice(df.index, 10), 'temperatura'] = np.nan

media = df['temperatura'].mean()
mediana = df['temperatura'].median()
desviacion = df['temperatura'].std()

df['temperatura'] = df['temperatura'].fillna(
    media
)

correlacion = df['temperatura'].corr(
    df['tasa_error'],
    method='pearson'
)

promedio_turno = df.groupby('turno')['tasa_error'].mean()

from scipy.stats import ttest_ind

matutino = df[df['turno'] == 'Matutino']['temperatura']
vespertino = df[df['turno'] == 'Vespertino']['temperatura']

t_stat, p_valor = ttest_ind(
    matutino,
    vespertino
)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

sns.boxplot(
    x='turno',
    y='tasa_error',
    data=df
)

plt.title('Distribución de errores por turno')
plt.show()

plt.figure(figsize=(8,5))

sns.regplot(
    x='temperatura',
    y='tasa_error',
    data=df
)

plt.title('Temperatura vs Tasa de Error')
plt.show()