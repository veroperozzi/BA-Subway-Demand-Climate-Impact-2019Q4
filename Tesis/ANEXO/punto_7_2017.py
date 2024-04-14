import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('fechas_corregidas2017.csv')


# Cargamos los datos desde un archivo CSV
datos = pd.read_csv('fechas_corregidas2017.csv')

# Convertimos la columna 'FECHA' a datetime para poder manipularla más fácilmente
datos['FECHA'] = pd.to_datetime(datos['FECHA'])

# Graficamos los precios a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(datos['FECHA'], datos['PRECIO'], label='Precio')
plt.title('Cambio en el Precio a lo largo del año 2017')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calcular la varianza del Precio
print('Varianza del Precio')
varianza_total_precio = int(df['PRECIO'].var())
varianza_total_precio
print(varianza_total_precio)

print('Promedio del PRECIO')
promedio_total_precio = int(df['PRECIO'].mean())
promedio_total_precio
print(promedio_total_precio)


###############################################

# Graficamos los TOTALES a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(datos['FECHA'], datos['TOTAL'], label='Total')
plt.title('Demanda del Total de Pasajes, linea A a lo largo del año 2017')
plt.xlabel('Fecha')
plt.ylabel('TOTAL')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calcular la varianza del Precio
print('Varianza del TOTAL de pasajes')
varianza_total = int(df['TOTAL'].var())
varianza_total
print(varianza_total)

print('Promedio del Total')
promedio_total = int(df['TOTAL'].mean())
promedio_total
print(promedio_total)

###############################
#Correlacion entre TOTAL y Precio

# Calculamos la correlación entre 'TOTAL' y 'PRECIO'
correlacion = datos['TOTAL'].corr(datos['PRECIO'])
print(f"La correlación entre TOTAL y PRECIO es: {correlacion}")

####################################
#Grafico de regresion lineal entre TOTAL y Precio

# Seleccionamos las variables 'TOTAL' como variable independiente (X) y 'PRECIO' como variable dependiente (y)
X = df[['TOTAL']]
y = df['PRECIO']
# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Creamos un modelo de regresión lineal
modelo = LinearRegression()
# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
# Realizamos predicciones con los datos de prueba
y_pred = modelo.predict(X_test)

# Graficamos los resultados
plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regresión lineal')
plt.title('Regresión Lineal entre TOTAL (demanda pasajes) y PRECIO año 2017')
plt.xlabel('TOTAL')
plt.ylabel('PRECIO')
plt.legend()
plt.show()

# Mostramos el coeficiente e intercepto del modelo
print(f"Coeficiente: {modelo.coef_[0]}")
print(f"Intercepto: {modelo.intercept_}")