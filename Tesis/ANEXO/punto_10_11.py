#Importar librerias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Asumiendo que el tamaño del chunk es adecuado para leer el 60% del archivo
chunk_size = 60000  # Ajusta este valor según el tamaño de tu archivo
chunk_iterator = pd.read_csv('./molinetes/precipitaciones_2016_limpio.csv', dayfirst=True, chunksize=chunk_size)

# Leer y concatenar los chunks necesarios para obtener aproximadamente el 60% del archivo
df_chunks = [chunk for _, chunk in zip(range(4), chunk_iterator)]  # Ajusta el rango para obtener el 60%
df = pd.concat(df_chunks)

# Convertir 'FECHA' a datetime y extraer características relevantes
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')
df['dia_semana'] = df['FECHA'].dt.dayofweek
df['es_fin_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

# Codificación de variables categóricas y preparación del conjunto de datos
df = pd.get_dummies(df, columns=['LINEA', 'ESTACION', 'MOLINETE'], drop_first=True)
X = df.drop(['FECHA', 'DESDE', 'HASTA', 'TOTAL'], axis=1)  # Asume que estas columnas no son relevantes o duplicadas
y = df['TOTAL']

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo RandomForestRegressor
clf = RandomForestRegressor(random_state=42)
clf.fit(X_train, y_train)

# Evaluación de la importancia de las características
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Seleccionar dinámicamente características basadas en la importancia (umbral del 0.1% de importancia)
threshold = 0.001  # 0.1%
features_to_keep = feature_importances[feature_importances > threshold].index.tolist()

# Gráfico de la importancia de las características seleccionadas
plt.figure(figsize=(10, 6))
feature_importances[features_to_keep].plot(kind='bar')
plt.title('Importancia de las Características Seleccionadas')
plt.xlabel('Características')
plt.ylabel('Importancia (%)')
plt.show()

# Correlación entre las características seleccionadas y la variable objetivo
X_selected = X_train[features_to_keep]
X_selected['TOTAL'] = y_train  # Añadir temporalmente la variable objetivo para calcular la correlación
correlation_matrix = X_selected.corr()
print("Correlación entre características seleccionadas y TOTAL:\n", correlation_matrix['TOTAL'])

# Remover la variable objetivo del DataFrame después de calcular la correlación
X_selected = X_selected.drop(columns=['TOTAL'])

# Gráficos de Dispersión para las relaciones entre características seleccionadas y la variable objetivo
fig, axs = plt.subplots(len(features_to_keep), figsize=(10, 5 * len(features_to_keep)))
fig.suptitle('Relación entre Características Seleccionadas y TOTAL')

for ax, feature in zip(axs, features_to_keep):
    if len(features_to_keep) > 1:
        ax.scatter(X_selected[feature], y_train, alpha=0.5)
    else:  # Ajuste para cuando hay solo una característica seleccionada
        axs.scatter(X_selected[feature], y_train, alpha=0.5)
    ax.set_title(f'{feature} vs. TOTAL')
    ax.set_xlabel(feature)
    ax.set_ylabel('TOTAL')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
