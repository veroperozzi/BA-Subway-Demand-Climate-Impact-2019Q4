import pandas as pd
from matplotlib import pyplot as plt, colors

# Cargar el archivo CSV
df = pd.read_csv('molinetes_2016_2017_2018_2019_concatenados_V1.0.csv')

pax_pagos_Total = 0.0
pax_pagos_NPrecip_Nferi_Nsemana = 0
pax_pagos_NPrecip_Nferi_semana = 0
pax_pagos_NPrecip_feri_Nsemana = 0
pax_pagos_NPrecip_feri_semana = 0
pax_pagos_Precip_Nferi_Nsemana = 0
pax_pagos_Precip_Nferi_semana = 0
pax_pagos_Precip_feri_Nsemana = 0
pax_pagos_Precip_feri_semana = 0

pax_pases_pagos_Total = 0.0
pax_pases_pagos_NPrecip_Nferi_Nsemana = 0
pax_pases_pagos_NPrecip_Nferi_semana = 0
pax_pases_pagos_NPrecip_feri_Nsemana = 0
pax_pases_pagos_NPrecip_feri_semana = 0
pax_pases_pagos_Precip_Nferi_Nsemana = 0
pax_pases_pagos_Precip_Nferi_semana = 0
pax_pases_pagos_Precip_feri_Nsemana = 0
pax_pases_pagos_Precip_feri_semana = 0

pax_freq_Total = 0.0
pax_freq_NPrecip_Nferi_Nsemana = 0
pax_freq_NPrecip_Nferi_semana = 0
pax_freq_NPrecip_feri_Nsemana = 0
pax_freq_NPrecip_feri_semana = 0
pax_freq_Precip_Nferi_Nsemana = 0
pax_freq_Precip_Nferi_semana = 0
pax_freq_Precip_feri_Nsemana = 0
pax_freq_Precip_feri_semana = 0

# Iterar por todas las filas del DataFrame
for index, row in df.iterrows():
    fecha = pd.to_datetime(row['FECHA'])
    # if fecha.year == 2016:
    pax_pagos = int(row['PAX_PAGOS'])
    pax_pases_pagos = int(row['PAX_PASES_PAGOS'])
    pax_freq = int(row['PAX_FREQ'])
    if row['PAX_PAGOS'] > 0:
        pax_pagos_Total = pax_pagos_Total + pax_pagos
        # Pregunto por no lluvio, no feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_pagos_NPrecip_Nferi_Nsemana = pax_pagos_NPrecip_Nferi_Nsemana + pax_pagos
        # Pregunto por no lluvio, no feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_pagos_NPrecip_Nferi_semana = pax_pagos_NPrecip_Nferi_semana + pax_pagos
        # Pregunto por no lluvio, feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_pagos_NPrecip_feri_Nsemana = pax_pagos_NPrecip_feri_Nsemana + pax_pagos
        # Pregunto por no lluvia, feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_pagos_NPrecip_feri_semana = pax_pagos_NPrecip_feri_semana + pax_pagos
        # Pregunto por lluvia, no feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_pagos_Precip_Nferi_Nsemana = pax_pagos_Precip_Nferi_Nsemana + pax_pagos
            # Pregunto por lluvia, no feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_pagos_Precip_Nferi_semana = pax_pagos_Precip_Nferi_semana + pax_pagos
        # Pregunto por la lluvia, feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_pagos_Precip_feri_Nsemana = pax_pagos_Precip_feri_Nsemana + pax_pagos
        # Pregunto por la lluvia, feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_pagos_Precip_feri_semana = pax_pagos_Precip_feri_semana + pax_pagos

    if row['PAX_PASES_PAGOS'] > 0:
        pax_pases_pagos_Total = pax_pases_pagos_Total + pax_pases_pagos
        # Pregunto por no lluvio, no feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_pases_pagos_NPrecip_Nferi_Nsemana = pax_pases_pagos_NPrecip_Nferi_Nsemana + pax_pases_pagos
        # Pregunto por no lluvio, no feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_pases_pagos_NPrecip_Nferi_semana = pax_pases_pagos_NPrecip_Nferi_semana + pax_pases_pagos
        # Pregunto por no lluvio, feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_pases_pagos_NPrecip_feri_Nsemana = pax_pases_pagos_NPrecip_feri_Nsemana + pax_pases_pagos
        # Pregunto por no lluvia, feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_pases_pagos_NPrecip_feri_semana = pax_pases_pagos_NPrecip_feri_semana + pax_pases_pagos
        # Pregunto por lluvia, no feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_pases_pagos_Precip_Nferi_Nsemana = pax_pases_pagos_Precip_Nferi_Nsemana + pax_pases_pagos
            # Pregunto por lluvia, no feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_pases_pagos_Precip_Nferi_semana = pax_pases_pagos_Precip_Nferi_semana + pax_pases_pagos
        # Pregunto por la lluvia, feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_pases_pagos_Precip_feri_Nsemana = pax_pases_pagos_Precip_feri_Nsemana + pax_pases_pagos
        # Pregunto por la lluvia, feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_pases_pagos_Precip_feri_semana = pax_pases_pagos_Precip_feri_semana + pax_pases_pagos

    if row['PAX_FREQ'] > 0:
        pax_freq_Total = pax_freq_Total + pax_freq
        # Pregunto por no lluvio, no feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_freq_NPrecip_Nferi_Nsemana = pax_freq_NPrecip_Nferi_Nsemana + pax_freq
        # Pregunto por no lluvio, no feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_freq_NPrecip_Nferi_semana = pax_freq_NPrecip_Nferi_semana + pax_freq
        # Pregunto por no lluvio, feriado y no dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_freq_NPrecip_feri_Nsemana = pax_freq_NPrecip_feri_Nsemana + pax_freq
        # Pregunto por no lluvia, feriado y dia de samana
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_freq_NPrecip_feri_semana = pax_freq_NPrecip_feri_semana + pax_freq
        # Pregunto por lluvia, no feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5:
            pax_freq_Precip_Nferi_Nsemana = pax_freq_Precip_Nferi_Nsemana + pax_freq
            # Pregunto por lluvia, no feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5:
            pax_freq_Precip_Nferi_semana = pax_freq_Precip_Nferi_semana + pax_freq
        # Pregunto por la lluvia, feriado y no dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5:
            pax_freq_Precip_feri_Nsemana = pax_freq_Precip_feri_Nsemana + pax_freq
        # Pregunto por la lluvia, feriado y dia de samana
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5:
            pax_freq_Precip_feri_semana = pax_freq_Precip_feri_semana + pax_freq

total_demanda = pax_freq_Total + pax_pagos_Total + pax_pases_pagos_Total
print(total_demanda)
print('###########PAX_PAGOS##############')
# print(f"Pagos sin precipitación, no feriado, fin de semana: {pax_pagos_NPrecip_Nferi_Nsemana}")
# print(f"Pagos sin precipitación, no feriado, día de semana: {pax_pagos_NPrecip_Nferi_semana}")
# print(f"Pagos sin precipitación, feriado, fin de semana: {pax_pagos_NPrecip_feri_Nsemana}")
# print(f"Pagos sin precipitación, feriado, día de semana: {pax_pagos_NPrecip_feri_semana}")
# print(f"Pagos con precipitación, no feriado, fin de semana: {pax_pagos_Precip_Nferi_Nsemana}")
# print(f"Pagos con precipitación, no feriado, día de semana: {pax_pagos_Precip_Nferi_semana}")
# print(f"Pagos con precipitación, feriado, fin de semana: {pax_pagos_Precip_feri_Nsemana}")
# print(f"Pagos con precipitación, feriado, día de semana: {pax_pagos_Precip_feri_semana}")

print(
    f"pax_pagos sin precipitación, no feriado, fin de semana: {pax_pagos_NPrecip_Nferi_Nsemana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, no feriado, día de semana: {pax_pagos_NPrecip_Nferi_semana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, feriado, fin de semana: {pax_pagos_NPrecip_feri_Nsemana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, feriado, día de semana: {pax_pagos_NPrecip_feri_semana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos con precipitación, no feriado, fin de semana: {pax_pagos_Precip_Nferi_Nsemana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos con precipitación, no feriado, día de semana: {pax_pagos_Precip_Nferi_semana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos con precipitación, feriado, fin de semana: {pax_pagos_Precip_feri_Nsemana / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos con precipitación, feriado, día de semana: {pax_pagos_Precip_feri_semana / pax_pagos_Total * 100:.2f}%")

print('###########PAX_PASES_PAGOS##############')
# print(f"Pagos sin precipitación, no feriado, fin de semana: {pax_pases_pagos_NPrecip_Nferi_Nsemana}")
# print(f"Pagos sin precipitación, no feriado, día de semana: {pax_pases_pagos_NPrecip_Nferi_semana}")
# print(f"Pagos sin precipitación, feriado, fin de semana: {pax_pases_pagos_NPrecip_feri_Nsemana}")
# print(f"Pagos sin precipitación, feriado, día de semana: {pax_pases_pagos_NPrecip_feri_semana}")
# print(f"Pagos con precipitación, no feriado, fin de semana: {pax_pases_pagos_Precip_Nferi_Nsemana}")
# print(f"Pagos con precipitación, no feriado, día de semana: {pax_pases_pagos_Precip_Nferi_semana}")
# print(f"Pagos con precipitación, feriado, fin de semana: {pax_pases_pagos_Precip_feri_Nsemana}")
# print(f"Pagos con precipitación, feriado, día de semana: {pax_pases_pagos_Precip_feri_semana}")

print(
    f"pax_pases_pagos sin precipitación, no feriado, fin de semana: {pax_pases_pagos_NPrecip_Nferi_Nsemana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, no feriado, día de semana: {pax_pases_pagos_NPrecip_Nferi_semana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, feriado, fin de semana: {pax_pases_pagos_NPrecip_feri_Nsemana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, feriado, día de semana: {pax_pases_pagos_NPrecip_feri_semana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, no feriado, fin de semana: {pax_pases_pagos_Precip_Nferi_Nsemana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, no feriado, día de semana: {pax_pases_pagos_Precip_Nferi_semana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, feriado, fin de semana: {pax_pases_pagos_Precip_feri_Nsemana / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, feriado, día de semana: {pax_pases_pagos_Precip_feri_semana / pax_pases_pagos_Total * 100:.2f}%")

print('###########PAX_FREQ_PAGOS##############')
# print(f"Pagos sin precipitación, no feriado, fin de semana: {pax_freq_NPrecip_Nferi_Nsemana}")
# print(f"Pagos sin precipitación, no feriado, día de semana: {pax_freq_NPrecip_Nferi_semana}")
# print(f"Pagos sin precipitación, feriado, fin de semana: {pax_freq_NPrecip_feri_Nsemana}")
# print(f"Pagos sin precipitación, feriado, día de semana: {pax_freq_NPrecip_feri_semana}")
# print(f"Pagos con precipitación, no feriado, fin de semana: {pax_freq_Precip_Nferi_Nsemana}")
# print(f"Pagos con precipitación, no feriado, día de semana: {pax_freq_Precip_Nferi_semana}")
# print(f"Pagos con precipitación, feriado, fin de semana: {pax_freq_Precip_feri_Nsemana}")
# print(f"Pagos con precipitación, feriado, día de semana: {pax_freq_Precip_feri_semana}")

print(
    f"pax_freq_pagos sin precipitación, no feriado, fin de semana: {pax_freq_NPrecip_Nferi_Nsemana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, no feriado, día de semana: {pax_freq_NPrecip_Nferi_semana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, feriado, fin de semana: {pax_freq_NPrecip_feri_Nsemana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, feriado, día de semana: {pax_freq_NPrecip_feri_semana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, no feriado, fin de semana: {pax_freq_Precip_Nferi_Nsemana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, no feriado, día de semana: {pax_freq_Precip_Nferi_semana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, feriado, fin de semana: {pax_freq_Precip_feri_Nsemana / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, feriado, día de semana: {pax_freq_Precip_feri_semana / pax_freq_Total * 100:.2f}%")

print('###########TOTAL_DEMANDA##############')
print('###########PAX_PAGOS##############')

print(
    f"pax_pagos sin precipitación, no feriado, fin de semana: {pax_pagos_NPrecip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, no feriado, día de semana: {pax_pagos_NPrecip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, feriado, fin de semana: {pax_pagos_NPrecip_feri_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos sin precipitación, feriado, día de semana: {pax_pagos_NPrecip_feri_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos con precipitación, no feriado, fin de semana: {pax_pagos_Precip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos con precipitación, no feriado, día de semana: {pax_pagos_Precip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pagos con precipitación, feriado, fin de semana: {pax_pagos_Precip_feri_Nsemana / total_demanda * 100:.2f}%")
print(f"pax_pagos con precipitación, feriado, día de semana: {pax_pagos_Precip_feri_semana / total_demanda * 100:.2f}%")

print('###########PAX_PASES_PAGOS##############')

print(
    f"pax_pases_pagos sin precipitación, no feriado, fin de semana: {pax_pases_pagos_NPrecip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, no feriado, día de semana: {pax_pases_pagos_NPrecip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, feriado, fin de semana: {pax_pases_pagos_NPrecip_feri_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos sin precipitación, feriado, día de semana: {pax_pases_pagos_NPrecip_feri_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, no feriado, fin de semana: {pax_pases_pagos_Precip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, no feriado, día de semana: {pax_pases_pagos_Precip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, feriado, fin de semana: {pax_pases_pagos_Precip_feri_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos con precipitación, feriado, día de semana: {pax_pases_pagos_Precip_feri_semana / total_demanda * 100:.2f}%")

print('###########PAX_FREQ_PAGOS##############')

print(
    f"pax_freq_pagos sin precipitación, no feriado, fin de semana: {pax_freq_NPrecip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, no feriado, día de semana: {pax_freq_NPrecip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, feriado, fin de semana: {pax_freq_NPrecip_feri_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos sin precipitación, feriado, día de semana: {pax_freq_NPrecip_feri_semana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, no feriado, fin de semana: {pax_freq_Precip_Nferi_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, no feriado, día de semana: {pax_freq_Precip_Nferi_semana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, feriado, fin de semana: {pax_freq_Precip_feri_Nsemana / total_demanda * 100:.2f}%")
print(
    f"pax_freq_pagos con precipitación, feriado, día de semana: {pax_freq_Precip_feri_semana / total_demanda * 100:.2f}%")

# GRAFICOS
##Graficos por grupo de demanda
# Datos actualizados para los tres grupos de demanda
import matplotlib.pyplot as plt

# Datos actualizados para los tres grupos de demanda
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

# Reajustando la configuración para ubicar los porcentajes fuera de los gráficos correctamente
fig, ax = plt.subplots(1, 4, figsize=(30, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.5]})

# Colores para los gráficos de torta
colors = plt.get_cmap('tab20').colors

# Etiquetas de porcentaje formateadas
porcentajes_labels_pax_pagos = ['{:.2f}%'.format(pct) for pct in porcentajes_pax_pagos]
porcentajes_labels_pax_pases_pagos = ['{:.2f}%'.format(pct) for pct in porcentajes_pax_pases_pagos]
porcentajes_labels_pax_freq_pagos = ['{:.2f}%'.format(pct) for pct in porcentajes_pax_freq_pagos]

# Configurando los gráficos de torta con porcentajes fuera
for i, (porcentajes, labels) in enumerate([
    (porcentajes_pax_pagos, porcentajes_labels_pax_pagos),
    (porcentajes_pax_pases_pagos, porcentajes_labels_pax_pases_pagos),
    (porcentajes_pax_freq_pagos, porcentajes_labels_pax_freq_pagos)]):

    wedges, texts, autotexts = ax[i].pie(
        porcentajes, labels=labels, startangle=140, colors=colors,
        autopct='', pctdistance=1.35)
    ax[i].set_title(['PAX_PAGOS', 'PAX_PASES_PAGOS', 'PAX_FREQ_PAGOS'][i])
    ax[i].axis('equal')

# Eliminando el eje del cuarto panel y agregando la leyenda
ax[3].axis('off')
ax[3].legend(wedges, categorias, title="Categorías", loc="center", prop={'size': 9})

plt.tight_layout()
plt.show()

# Ahora, los gráficos de barras para cada grupo, para una comparación detallada
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# PAX_PAGOS
axes[0].barh(categorias, porcentajes_pax_pagos, color='lightblue')
axes[0].set_title('PAX_PAGOS')
axes[0].set_xlabel('Porcentaje')

# PAX_PASES_PAGOS
axes[1].barh(categorias, porcentajes_pax_pases_pagos, color='lightgreen')
axes[1].set_title('PAX_PASES_PAGOS')
axes[1].set_xlabel('Porcentaje')

# PAX_FREQ_PAGOS
axes[2].barh(categorias, porcentajes_pax_freq_pagos, color='salmon')
axes[2].set_title('PAX_FREQ_PAGOS')
axes[2].set_xlabel('Porcentaje')

plt.tight_layout()
plt.show()

# GRAFICOS
##Graficos tomando en cuenta demanda total
# Datos sumados para cada tipo de pasajero
from matplotlib import pyplot as plt

total_pax_pagos = sum([2.62, 59.44, 0.13, 1.67, 0.84, 23.36, 0.05, 0.79])
total_pax_pases_pagos = sum([0.00, 0.32, 0.00, 0.01, 0.00, 0.13, 0.00, 0.00])
total_pax_freq_pagos = sum([0.10, 2.20, 0.01, 0.06, 0.03, 0.86, 0.00, 0.04])

# Datos para el gráfico de torta
sizes = [total_pax_pagos, total_pax_pases_pagos, total_pax_freq_pagos]
labels = ['Pax Pagos', 'Pax Pases Pagos', 'Pax Freq Pagos']
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.1, 0, 0)  # solo "explotar" el primer trozo (i.e., 'Pax Pagos')

# Utilizando una función lambda para aumentar el espacio para los porcentajes
autopct = lambda pct: f'{pct:.1f}%' if pct > 0 else ''

# Ajustando el parámetro 'explode' para mejorar la visualización
explode = (0.1, 0.3, 0.2)  # Separando todas las secciones para mejor claridad

# Creando el gráfico de torta nuevamente con los ajustes
fig1, ax1 = plt.subplots()
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=autopct,
                                   shadow=True, startangle=90)

# Mejorando la visibilidad de los porcentajes moviéndolos más hacia afuera
for autotext in autotexts:
    autotext.set_color('black')
plt.setp(autotexts, size=8, weight="bold")
ax1.axis('equal')

plt.title('Distribución de la demanda por tipo de pasajero')
plt.show()