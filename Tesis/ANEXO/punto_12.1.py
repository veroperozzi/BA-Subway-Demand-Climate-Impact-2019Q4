import matplotlib.pyplot as plt

# Datos originales
categorias = [
    "Sin precipitación, no feriado, fin de semana",
    "Sin precipitación, no feriado, día de semana",
    "Sin precipitación, feriado, fin de semana",
    "Sin precipitación, feriado, día de semana",
    "Con precipitación, no feriado, fin de semana",
    "Con precipitación, no feriado, día de semana",
    "Con precipitación, feriado, fin de semana",
    "Con precipitación, feriado, día de semana"
]

# Porcentajes para cada grupo de demanda
porcentajes_pax_pagos = [2.72, 61.92, 0.13, 1.74, 0.87, 24.33, 0.05, 0.82]
porcentajes_pax_pases_pagos = [0.19, 68.02, 0.01, 1.59, 0.06, 26.67, 0.00, 1.02]
porcentajes_pax_freq_pagos = [2.72, 62.46, 0.15, 1.65, 0.94, 24.33, 0.05, 1.05]

# Calculando nuevos porcentajes para "Con precipitación" y "Sin precipitación"
nuevos_porcentajes = lambda porcentajes: [
    sum(porcentajes[:4]),  # Sin precipitación
    sum(porcentajes[4:])   # Con precipitación
]

# Aplicando el cálculo a cada grupo de demanda
nuevos_porcentajes_pax_pagos = [70.22, 29.78]
nuevos_porcentajes_pax_pases_pagos = [71.03, 28.92]
nuevos_porcentajes_pax_freq_pagos = [70.31, 29.69]

# Nuevas categorías
nuevas_categorias = ['Sin precipitación', 'Con precipitación']

# Creando el nuevo gráfico
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Definiendo colores
colors = ['skyblue', 'lightgreen']

# Títulos para cada sub-gráfico
titulos = ['PAX_PAGOS', 'PAX_PASES_PAGOS', 'PAX_FREQ_PAGOS']

# Configuración de los gráficos
for i, porcentajes in enumerate([nuevos_porcentajes_pax_pagos, nuevos_porcentajes_pax_pases_pagos, nuevos_porcentajes_pax_freq_pagos]):
    ax[i].pie(porcentajes, labels=nuevas_categorias, autopct='%1.1f%%', startangle=140, colors=colors)
    ax[i].set_title(titulos[i])

plt.tight_layout()
plt.show()
