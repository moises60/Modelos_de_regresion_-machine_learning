import numpy as np
import pandas as pd

# Fijar la semilla para reproducibilidad
np.random.seed(42)
n_samples = 200

# Generación de variables independientes
TV = np.random.uniform(0, 300, n_samples)
Radio = np.random.uniform(0, 50, n_samples)
Newspaper = np.random.uniform(0, 100, n_samples)
Online = np.random.uniform(0, 150, n_samples)
SocialMedia = np.random.uniform(0, 200, n_samples)

# Generación de la variable dependiente 'Sales' con coeficientes ajustados y ruido reducido
Sales = (
    10 +
    0.5 * TV +         # Coeficiente aumentado
    1.5 * Radio +      # Coeficiente aumentado
    0.7 * Newspaper +
    0.4 * Online +
    0.6 * SocialMedia +
    np.random.normal(0, 5, n_samples)  # Ruido reducido
)

# Creación del DataFrame
data = {
    'TV': TV,
    'Radio': Radio,
    'Newspaper': Newspaper,
    'Online': Online,
    'SocialMedia': SocialMedia,
    'Sales': Sales
}

df = pd.DataFrame(data)

# Mostrar las primeras 5 filas del dataset
print("Primeras 5 filas del dataset sintético mejorado:")
print(df.head())

# Guardar el dataset en un archivo CSV
df.to_csv('Advertising_Synthetic_Data.csv', index=False)
