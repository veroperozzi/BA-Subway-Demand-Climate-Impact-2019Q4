import matplotlib.pyplot as plt

# Datos proporcionados
datos = {
    "PAX_PAGOS": {
        "no_llovio": 1.25 + 1.94 + 27.10 + 32.33 + 0.47 + 0.71,
        "llovio": 0.30 + 0.46 + 12.57 + 15.04 + 0.11 + 0.17
    },
    "PAX_PASES_PAGOS": {
        "no_llovio": 0.09 + 0.22 + 25.05 + 39.89 + 0.12 + 0.33,
        "llovio": 0.03 + 0.06 + 12.07 + 19.43 + 0.03 + 0.07
    },
    "PAX_FREQ_PAGOS": {
        "no_llovio": 0.83 + 1.98 + 22.38 + 38.86 + 0.31 + 0.74,
        "llovio": 0.21 + 0.49 + 10.28 + 17.86 + 0.06 + 0.15
    }
}

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Generar gráficos de torta
for i, (tipo, valores) in enumerate(datos.items()):
    axs[i].pie(valores.values(), labels=valores.keys(), autopct='%1.1f%%', startangle=140)
    axs[i].set_title(tipo)

plt.tight_layout()
plt.show()

# Corrección de la interpretación para ajustarse a la nueva instrucción
datos_corregidos = {
    "PAX_PAGOS": {
        "Centrico (No Llueve)": 1.25 + 27.10 + 0.47,  # Fin de semana + Día de semana + Feriado
        "Residencial (No Llueve)": 1.94 + 32.33 + 0.71,
        "Centrico (Llueve)": 0.30 + 12.57 + 0.11,
        "Residencial (Llueve)": 0.46 + 15.04 + 0.17
    },
    "PAX_PASES_PAGOS": {
        "Centrico (No Llueve)": 0.09 + 25.05 + 0.12,
        "Residencial (No Llueve)": 0.22 + 39.89 + 0.33,
        "Centrico (Llueve)": 0.03 + 12.07 + 0.03,
        "Residencial (Llueve)": 0.06 + 19.43 + 0.07
    },
    "PAX_FREQ_PAGOS": {
        "Centrico (No Llueve)": 0.83 + 22.38 + 0.31,
        "Residencial (No Llueve)": 1.98 + 38.86 + 0.74,
        "Centrico (Llueve)": 0.21 + 10.28 + 0.06,
        "Residencial (Llueve)": 0.49 + 17.86 + 0.15
    }
}

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Generar gráficos de torta con la corrección
for i, (tipo, valores) in enumerate(datos_corregidos.items()):
    axs[i].pie(valores.values(), labels=valores.keys(), autopct='%1.1f%%', startangle=140)
    axs[i].set_title(tipo)

plt.tight_layout()
plt.show()

