# 1. Importar las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import os

# Crear una carpeta llamada 'graficas' si no existe
if not os.path.exists('graficas'):
    os.makedirs('graficas')

# Fijar la semilla para reproducibilidad
random_state = 42

# 2. Cargar el dataset
df = pd.read_csv('Advertising_Synthetic_Data.csv')

# 3. Exploración de Datos
print("\nInformación del dataset:")
print(df.info())

print("\nDescripción estadística:")
print(df.describe())

# Matriz de correlación
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Matriz de Correlación - Dataset")
plt.savefig('graficas/matriz_correlacion.png', dpi=300, bbox_inches='tight')  # Guardar la gráfica
plt.show()

# Análisis de multicolinealidad mediante VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]
    return vif_data

X = df[['TV', 'Radio', 'Newspaper', 'Online', 'SocialMedia']]
y = df['Sales']

print("\nFactor de Inflación de la Varianza (VIF):")
print(calculate_vif(X))

# 4. División de Datos en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# 5. Definición de una función para imprimir métricas
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# 6. Entrenamiento de Modelos con Pipelines

# Regresión Lineal
pipeline_lin = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_lin.fit(X_train, y_train)
y_pred_lin = pipeline_lin.predict(X_test)

# Regresión Ridge
pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0, random_state=random_state))
])
pipeline_ridge.fit(X_train, y_train)
y_pred_ridge = pipeline_ridge.predict(X_test)

# Regresión Lasso
pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=0.1, max_iter=10000, random_state=random_state))
])
pipeline_lasso.fit(X_train, y_train)
y_pred_lasso = pipeline_lasso.predict(X_test)

# Regresión ElasticNet
pipeline_elastic = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ElasticNet(random_state=random_state, max_iter=10000))
])
pipeline_elastic.fit(X_train, y_train)
y_pred_elastic = pipeline_elastic.predict(X_test)

# 7. Evaluación de Modelos
print("\nEvaluación de Modelos:")
print_metrics(y_test, y_pred_lin, "Regresión Lineal")
print_metrics(y_test, y_pred_ridge, "Regresión Ridge")
print_metrics(y_test, y_pred_lasso, "Regresión Lasso")
print_metrics(y_test, y_pred_elastic, "Regresión ElasticNet")

# 8. Validación Cruzada con Pipelines
cv_scores_lin = cross_val_score(pipeline_lin, X, y, cv=5, scoring='r2')
print(f"\nRegresión Lineal - R² promedio en 5-fold CV: {cv_scores_lin.mean():.2f}")

cv_scores_ridge = cross_val_score(pipeline_ridge, X, y, cv=5, scoring='r2')
print(f"Regresión Ridge - R² promedio en 5-fold CV: {cv_scores_ridge.mean():.2f}")

cv_scores_lasso = cross_val_score(pipeline_lasso, X, y, cv=5, scoring='r2')
print(f"Regresión Lasso - R² promedio en 5-fold CV: {cv_scores_lasso.mean():.2f}")

cv_scores_elastic = cross_val_score(pipeline_elastic, X, y, cv=5, scoring='r2')
print(f"Regresión ElasticNet - R² promedio en 5-fold CV: {cv_scores_elastic.mean():.2f}")

# 9. Regularización con Grid Search

alphas = np.logspace(-3, 3, 100)

# Grid Search para Ridge
ridge_grid = GridSearchCV(
    estimator=pipeline_ridge,
    param_grid={'regressor__alpha': alphas},
    scoring='r2',
    cv=5
)
ridge_grid.fit(X_train, y_train)
print(f"\nMejor alpha para Ridge: {ridge_grid.best_params_['regressor__alpha']}")

# Grid Search para Lasso
lasso_grid = GridSearchCV(
    estimator=pipeline_lasso,
    param_grid={'regressor__alpha': alphas},
    scoring='r2',
    cv=5
)
lasso_grid.fit(X_train, y_train)
print(f"Mejor alpha para Lasso: {lasso_grid.best_params_['regressor__alpha']}")

# Grid Search para ElasticNet
param_grid_elastic = {
    'regressor__alpha': alphas,
    'regressor__l1_ratio': np.linspace(0, 1, 10)
}
elastic_grid = GridSearchCV(
    estimator=pipeline_elastic,
    param_grid=param_grid_elastic,
    scoring='r2',
    cv=5
)
elastic_grid.fit(X_train, y_train)
print(f"Mejores parámetros para ElasticNet: {elastic_grid.best_params_}")

# 10. Entrenar Modelos con los Mejores Parámetros

# Mejor Ridge
pipeline_best_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(
        alpha=ridge_grid.best_params_['regressor__alpha'],
        random_state=random_state
    ))
])
pipeline_best_ridge.fit(X_train, y_train)
y_pred_best_ridge = pipeline_best_ridge.predict(X_test)
print_metrics(y_test, y_pred_best_ridge, "Mejor Regresión Ridge")

# Mejor Lasso
pipeline_best_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(
        alpha=lasso_grid.best_params_['regressor__alpha'],
        max_iter=10000,
        random_state=random_state
    ))
])
pipeline_best_lasso.fit(X_train, y_train)
y_pred_best_lasso = pipeline_best_lasso.predict(X_test)
print_metrics(y_test, y_pred_best_lasso, "Mejor Regresión Lasso")

# Mejor ElasticNet
pipeline_best_elastic = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ElasticNet(
        alpha=elastic_grid.best_params_['regressor__alpha'],
        l1_ratio=elastic_grid.best_params_['regressor__l1_ratio'],
        max_iter=10000,
        random_state=random_state
    ))
])
pipeline_best_elastic.fit(X_train, y_train)
y_pred_best_elastic = pipeline_best_elastic.predict(X_test)
print_metrics(y_test, y_pred_best_elastic, "Mejor Regresión ElasticNet")

# 11. Regresión Lineal con Características Polinomiales

pipeline_poly = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_poly.fit(X_train, y_train)
y_pred_poly = pipeline_poly.predict(X_test)
print_metrics(y_test, y_pred_poly, "Regresión Lineal con Características Polinomiales")

# 12. Selección de Características con LassoCV

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=random_state)
lasso_cv.fit(X_train, y_train)

# Selección de características
model = SelectFromModel(lasso_cv, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# Entrenar un nuevo modelo con las características seleccionadas
regressor = LinearRegression()
regressor.fit(X_train_selected, y_train)
y_pred_selected = regressor.predict(X_test_selected)
print_metrics(y_test, y_pred_selected, "Regresión Lineal con Selección de Características")

# 13. Visualización de Resultados

# Comparación de Predicciones vs Valores Reales
comparison = pd.DataFrame({
    'Real': y_test,
    'Linear': y_pred_lin,
    'Ridge': y_pred_best_ridge,
    'Lasso': y_pred_best_lasso,
    'ElasticNet': y_pred_best_elastic,
    'Poly': y_pred_poly,
    'Selected Features': y_pred_selected
})

print("\nComparación de Predicciones vs Valores Reales:")
print(comparison.head())

# Visualización de Predicciones vs Valores Reales
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_lin, color='blue', label='Regresión Lineal', alpha=0.6)
plt.scatter(y_test, y_pred_best_ridge, color='green', label='Mejor Regresión Ridge', alpha=0.6)
plt.scatter(y_test, y_pred_best_lasso, color='red', label='Mejor Regresión Lasso', alpha=0.6)
plt.scatter(y_test, y_pred_best_elastic, color='purple', label='Mejor ElasticNet', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.savefig('graficas/predicciones_vs_reales.png', dpi=300, bbox_inches='tight')  # Guardar la gráfica
plt.show()

# 14. Análisis de Residuos

# Función para graficar y guardar residuos
def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    
    # Histograma de Residuos
    plt.figure(figsize=(10,6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Distribución de Residuos - {model_name}')
    plt.xlabel('Residuos')
    plt.savefig(f'graficas/distribucion_residuos_{model_name}.png', dpi=300, bbox_inches='tight')  # Guardar
    plt.show()
    
    # QQ-Plot de Residuos
    plt.figure(figsize=(10,6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'QQ-Plot de Residuos - {model_name}')
    plt.savefig(f'graficas/qqplot_residuos_{model_name}.png', dpi=300, bbox_inches='tight')  # Guardar
    plt.show()
    
    # Residuos vs Predicciones
    plt.figure(figsize=(10,6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title(f'Residuos vs Predicciones - {model_name}')
    plt.savefig(f'graficas/residuos_vs_predicciones_{model_name}.png', dpi=300, bbox_inches='tight')  # Guardar
    plt.show()

# Análisis de Residuos para cada modelo
plot_residuals(y_test, y_pred_lin, "Regresión Lineal")
plot_residuals(y_test, y_pred_best_ridge, "Mejor Regresión Ridge")
plot_residuals(y_test, y_pred_best_lasso, "Mejor Regresión Lasso")
plot_residuals(y_test, y_pred_best_elastic, "Mejor Regresión ElasticNet")
plot_residuals(y_test, y_pred_poly, "Regresión Polinomial")
plot_residuals(y_test, y_pred_selected, "Regresión con Selección de Características")

