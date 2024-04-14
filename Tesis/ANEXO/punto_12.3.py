import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('molinetes_2017_con_zonas.csv')

pax_pagos_Total = 0.0
pax_pagos_NPrecip_Nferi_Nsemana_NResidencial = 0
pax_pagos_NPrecip_Nferi_Nsemana_Residencial = 0
pax_pagos_NPrecip_Nferi_semana_NResidencial = 0
pax_pagos_NPrecip_Nferi_semana_Residencial = 0
pax_pagos_NPrecip_feri_Nsemana_NResidencial = 0
pax_pagos_NPrecip_feri_Nsemana_Residencial = 0
pax_pagos_NPrecip_feri_semana_NResidencial = 0
pax_pagos_NPrecip_feri_semana_Residencial = 0
pax_pagos_Precip_Nferi_Nsemana_NResidencial = 0
pax_pagos_Precip_Nferi_Nsemana_Residencial = 0
pax_pagos_Precip_Nferi_semana_NResidencial = 0
pax_pagos_Precip_Nferi_semana_Residencial = 0
pax_pagos_Precip_feri_Nsemana_NResidencial = 0
pax_pagos_Precip_feri_Nsemana_Residencial = 0
pax_pagos_Precip_feri_semana_NResidencial = 0
pax_pagos_Precip_feri_semana_Residencial = 0

pax_pases_pagos_Total = 0.0
pax_pases_pagos_NPrecip_Nferi_Nsemana_NResidencial = 0
pax_pases_pagos_NPrecip_Nferi_Nsemana_Residencial = 0
pax_pases_pagos_NPrecip_Nferi_semana_NResidencial = 0
pax_pases_pagos_NPrecip_Nferi_semana_Residencial = 0
pax_pases_pagos_NPrecip_feri_Nsemana_NResidencial = 0
pax_pases_pagos_NPrecip_feri_Nsemana_Residencial = 0
pax_pases_pagos_NPrecip_feri_semana_NResidencial = 0
pax_pases_pagos_NPrecip_feri_semana_Residencial = 0
pax_pases_pagos_Precip_Nferi_Nsemana_NResidencial = 0
pax_pases_pagos_Precip_Nferi_Nsemana_Residencial = 0
pax_pases_pagos_Precip_Nferi_semana_NResidencial = 0
pax_pases_pagos_Precip_Nferi_semana_Residencial = 0
pax_pases_pagos_Precip_feri_Nsemana_NResidencial = 0
pax_pases_pagos_Precip_feri_Nsemana_Residencial = 0
pax_pases_pagos_Precip_feri_semana_NResidencial = 0
pax_pases_pagos_Precip_feri_semana_Residencial = 0

pax_freq_Total = 0.0
pax_freq_NPrecip_Nferi_Nsemana_NResidencial = 0
pax_freq_NPrecip_Nferi_Nsemana_Residencial = 0
pax_freq_NPrecip_Nferi_semana_NResidencial = 0
pax_freq_NPrecip_Nferi_semana_Residencial = 0
pax_freq_NPrecip_feri_Nsemana_NResidencial = 0
pax_freq_NPrecip_feri_Nsemana_Residencial = 0
pax_freq_NPrecip_feri_semana_NResidencial = 0
pax_freq_NPrecip_feri_semana_Residencial = 0
pax_freq_Precip_Nferi_Nsemana_NResidencial = 0
pax_freq_Precip_Nferi_Nsemana_Residencial = 0
pax_freq_Precip_Nferi_semana_NResidencial = 0
pax_freq_Precip_Nferi_semana_Residencial = 0
pax_freq_Precip_feri_Nsemana_NResidencial = 0
pax_freq_Precip_feri_Nsemana_Residencial = 0
pax_freq_Precip_feri_semana_NResidencial = 0
pax_freq_Precip_feri_semana_Residencial = 0

# Iterar por todas las filas del DataFrame
for index, row in df.iterrows():
    fecha = pd.to_datetime(row['FECHA'])
    # if fecha.year == 2016:
    pax_pagos = int(row['PAX_PAGOS'])
    pax_pases_pagos = int(row['PAX_PASES_PAGOS'])
    pax_freq = int(row['PAX_FREQ'])
    if row['PAX_PAGOS'] > 0:
        pax_pagos_Total = pax_pagos_Total + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pagos_NPrecip_Nferi_Nsemana_NResidencial = pax_pagos_NPrecip_Nferi_Nsemana_NResidencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pagos_NPrecip_Nferi_Nsemana_Residencial = pax_pagos_NPrecip_Nferi_Nsemana_Residencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pagos_NPrecip_Nferi_semana_NResidencial = pax_pagos_NPrecip_Nferi_semana_NResidencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pagos_NPrecip_Nferi_semana_Residencial = pax_pagos_NPrecip_Nferi_semana_Residencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pagos_NPrecip_feri_Nsemana_NResidencial = pax_pagos_NPrecip_feri_Nsemana_NResidencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pagos_NPrecip_feri_Nsemana_Residencial = pax_pagos_NPrecip_feri_Nsemana_Residencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pagos_NPrecip_feri_semana_NResidencial = pax_pagos_NPrecip_feri_semana_NResidencial + pax_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pagos_NPrecip_feri_semana_Residencial = pax_pagos_NPrecip_feri_semana_Residencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pagos_Precip_Nferi_Nsemana_NResidencial = pax_pagos_Precip_Nferi_Nsemana_NResidencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pagos_Precip_Nferi_Nsemana_Residencial = pax_pagos_Precip_Nferi_Nsemana_Residencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pagos_Precip_Nferi_semana_NResidencial = pax_pagos_Precip_Nferi_semana_NResidencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pagos_Precip_Nferi_semana_Residencial = pax_pagos_Precip_Nferi_semana_Residencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pagos_Precip_feri_Nsemana_NResidencial = pax_pagos_Precip_feri_Nsemana_NResidencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pagos_Precip_feri_Nsemana_Residencial = pax_pagos_Precip_feri_Nsemana_Residencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pagos_Precip_feri_semana_NResidencial = pax_pagos_Precip_feri_semana_NResidencial + pax_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pagos_Precip_feri_semana_Residencial = pax_pagos_Precip_feri_semana_Residencial + pax_pagos

    if row['PAX_PASES_PAGOS'] > 0:
        pax_pases_pagos_Total = pax_pases_pagos_Total + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pases_pagos_NPrecip_Nferi_Nsemana_NResidencial = pax_pases_pagos_NPrecip_Nferi_Nsemana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pases_pagos_NPrecip_Nferi_Nsemana_Residencial = pax_pases_pagos_NPrecip_Nferi_Nsemana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pases_pagos_NPrecip_Nferi_semana_NResidencial = pax_pases_pagos_NPrecip_Nferi_semana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pases_pagos_NPrecip_Nferi_semana_Residencial = pax_pases_pagos_NPrecip_Nferi_semana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pases_pagos_NPrecip_feri_Nsemana_NResidencial = pax_pases_pagos_NPrecip_feri_Nsemana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pases_pagos_NPrecip_feri_Nsemana_Residencial = pax_pases_pagos_NPrecip_feri_Nsemana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pases_pagos_NPrecip_feri_semana_NResidencial = pax_pases_pagos_NPrecip_feri_semana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pases_pagos_NPrecip_feri_semana_Residencial = pax_pases_pagos_NPrecip_feri_semana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pases_pagos_Precip_Nferi_Nsemana_NResidencial = pax_pases_pagos_Precip_Nferi_Nsemana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pases_pagos_Precip_Nferi_Nsemana_Residencial = pax_pases_pagos_Precip_Nferi_Nsemana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pases_pagos_Precip_Nferi_semana_NResidencial = pax_pases_pagos_Precip_Nferi_semana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pases_pagos_Precip_Nferi_semana_Residencial = pax_pases_pagos_Precip_Nferi_semana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_pases_pagos_Precip_feri_Nsemana_NResidencial = pax_pases_pagos_Precip_feri_Nsemana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_pases_pagos_Precip_feri_Nsemana_Residencial = pax_pases_pagos_Precip_feri_Nsemana_Residencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_pases_pagos_Precip_feri_semana_NResidencial = pax_pases_pagos_Precip_feri_semana_NResidencial + pax_pases_pagos
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_pases_pagos_Precip_feri_semana_Residencial = pax_pases_pagos_Precip_feri_semana_Residencial + pax_pases_pagos

    if row['PAX_FREQ'] > 0:
        pax_freq_Total = pax_freq_Total + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_freq_NPrecip_Nferi_Nsemana_NResidencial = pax_freq_NPrecip_Nferi_Nsemana_NResidencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_freq_NPrecip_Nferi_Nsemana_Residencial = pax_freq_NPrecip_Nferi_Nsemana_Residencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_freq_NPrecip_Nferi_semana_NResidencial = pax_freq_NPrecip_Nferi_semana_NResidencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_freq_NPrecip_Nferi_semana_Residencial = pax_freq_NPrecip_Nferi_semana_Residencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_freq_NPrecip_feri_Nsemana_NResidencial = pax_freq_NPrecip_feri_Nsemana_NResidencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_freq_NPrecip_feri_Nsemana_Residencial = pax_freq_NPrecip_feri_Nsemana_Residencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_freq_NPrecip_feri_semana_NResidencial = pax_freq_NPrecip_feri_semana_NResidencial + pax_freq
        if row['PRECIPITACION'] <= 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_freq_NPrecip_feri_semana_Residencial = pax_freq_NPrecip_feri_semana_Residencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_freq_Precip_Nferi_Nsemana_NResidencial = pax_freq_Precip_Nferi_Nsemana_NResidencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_freq_Precip_Nferi_Nsemana_Residencial = pax_freq_Precip_Nferi_Nsemana_Residencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_freq_Precip_Nferi_semana_NResidencial = pax_freq_Precip_Nferi_semana_NResidencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 0 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_freq_Precip_Nferi_semana_Residencial = pax_freq_Precip_Nferi_semana_Residencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 0:
            pax_freq_Precip_feri_Nsemana_NResidencial = pax_freq_Precip_feri_Nsemana_NResidencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek > 5 and row['ZONA'] == 1:
            pax_freq_Precip_feri_Nsemana_Residencial = pax_freq_Precip_feri_Nsemana_Residencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 0:
            pax_freq_Precip_feri_semana_NResidencial = pax_freq_Precip_feri_semana_NResidencial + pax_freq
        if row['PRECIPITACION'] > 0.0 and row['FERIADO'] == 1 and fecha.dayofweek < 5 and row['ZONA'] == 1:
            pax_freq_Precip_feri_semana_Residencial = pax_freq_Precip_feri_semana_Residencial + pax_freq

total_demanda = pax_freq_Total + pax_pagos_Total + pax_pases_pagos_Total
print(total_demanda)
print('###########PAX_PAGOS##############')
print(
    f"pax_pagos no llovio, no feriado, fin de semana, centrico: {pax_pagos_NPrecip_Nferi_Nsemana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, fin de semana, residencial: {pax_pagos_NPrecip_Nferi_Nsemana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, dia de semana, centrico: {pax_pagos_NPrecip_Nferi_semana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, dia de semana, residencial: {pax_pagos_NPrecip_Nferi_semana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, fin de semana, centrico: {pax_pagos_NPrecip_feri_Nsemana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, fin de semana, residencial: {pax_pagos_NPrecip_feri_Nsemana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, dia de semana, centrico: {pax_pagos_NPrecip_feri_semana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, dia de semana, residencial: {pax_pagos_NPrecip_feri_semana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, fin de semana, centrico: {pax_pagos_Precip_Nferi_Nsemana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, fin de semana, residencial: {pax_pagos_Precip_Nferi_Nsemana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, dia de semana, centrico: {pax_pagos_Precip_Nferi_semana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, dia de semana, residencial: {pax_pagos_Precip_Nferi_semana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, fin de semana, centrico: {pax_pagos_Precip_feri_Nsemana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, fin de semana, residencial: {pax_pagos_Precip_feri_Nsemana_Residencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, dia de semana, centrico: {pax_pagos_Precip_feri_semana_NResidencial / pax_pagos_Total * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, dia de semana, residencial: {pax_pagos_Precip_feri_semana_Residencial / pax_pagos_Total * 100:.2f}%")

print('###########PAX_PASES_PAGOS##############')

print(
    f"pax_pases_pagos no llovio, no feriado, fin de semana, centrico: {pax_pases_pagos_NPrecip_Nferi_Nsemana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, fin de semana, residencial: {pax_pases_pagos_NPrecip_Nferi_Nsemana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, dia de semana, centrico: {pax_pases_pagos_NPrecip_Nferi_semana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, dia de semana, residencial: {pax_pases_pagos_NPrecip_Nferi_semana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, fin de semana, centrico: {pax_pases_pagos_NPrecip_feri_Nsemana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, fin de semana, residencial: {pax_pases_pagos_NPrecip_feri_Nsemana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, dia de semana, centrico: {pax_pases_pagos_NPrecip_feri_semana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, dia de semana, residencial: {pax_pases_pagos_NPrecip_feri_semana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, fin de semana, centrico: {pax_pases_pagos_Precip_Nferi_Nsemana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, fin de semana, residencial: {pax_pases_pagos_Precip_Nferi_Nsemana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, dia de semana, centrico: {pax_pases_pagos_Precip_Nferi_semana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, dia de semana, residencial: {pax_pases_pagos_Precip_Nferi_semana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, fin de semana, centrico: {pax_pases_pagos_Precip_feri_Nsemana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, fin de semana, residencial: {pax_pases_pagos_Precip_feri_Nsemana_Residencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, dia de semana, centrico: {pax_pases_pagos_Precip_feri_semana_NResidencial / pax_pases_pagos_Total * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, dia de semana, residencial: {pax_pases_pagos_Precip_feri_semana_Residencial / pax_pases_pagos_Total * 100:.2f}%")

print('###########PAX_FREQ_PAGOS##############')

print(
    f"pax_freq no llovio, no feriado, fin de semana, centrico: {pax_freq_NPrecip_Nferi_Nsemana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, fin de semana, residencial: {pax_freq_NPrecip_Nferi_Nsemana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, dia de semana, centrico: {pax_freq_NPrecip_Nferi_semana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, dia de semana, residencial: {pax_freq_NPrecip_Nferi_semana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, fin de semana, centrico: {pax_freq_NPrecip_feri_Nsemana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, fin de semana, residencial: {pax_freq_NPrecip_feri_Nsemana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, dia de semana, centrico: {pax_freq_NPrecip_feri_semana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, dia de semana, residencial: {pax_freq_NPrecip_feri_semana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, fin de semana, centrico: {pax_freq_Precip_Nferi_Nsemana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, fin de semana, residencial: {pax_freq_Precip_Nferi_Nsemana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, dia de semana, centrico: {pax_freq_Precip_Nferi_semana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, dia de semana, residencial: {pax_freq_Precip_Nferi_semana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, fin de semana, centrico: {pax_freq_Precip_feri_Nsemana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, fin de semana, residencial: {pax_freq_Precip_feri_Nsemana_Residencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, dia de semana, centrico: {pax_freq_Precip_feri_semana_NResidencial / pax_freq_Total * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, dia de semana, residencial: {pax_freq_Precip_feri_semana_Residencial / pax_freq_Total * 100:.2f}%")

print('###########TOTAL_DEMANDA##############')
print('###########PAX_PAGOS##############')

print(
    f"pax_pagos no llovio, no feriado, fin de semana, centrico: {pax_pagos_NPrecip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, fin de semana, residencial: {pax_pagos_NPrecip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, dia de semana, centrico: {pax_pagos_NPrecip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, no feriado, dia de semana, residencial: {pax_pagos_NPrecip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, fin de semana, centrico: {pax_pagos_NPrecip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, fin de semana, residencial: {pax_pagos_NPrecip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, dia de semana, centrico: {pax_pagos_NPrecip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos no llovio, feriado, dia de semana, residencial: {pax_pagos_NPrecip_feri_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, fin de semana, centrico: {pax_pagos_Precip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, fin de semana, residencial: {pax_pagos_Precip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, dia de semana, centrico: {pax_pagos_Precip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, no feriado, dia de semana, residencial: {pax_pagos_Precip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, fin de semana, centrico: {pax_pagos_Precip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, fin de semana, residencial: {pax_pagos_Precip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, dia de semana, centrico: {pax_pagos_Precip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pagos llovio, feriado, dia de semana, residencial: {pax_pagos_Precip_feri_semana_Residencial / total_demanda * 100:.2f}%")

print('###########PAX_PASES_PAGOS##############')

print(
    f"pax_pases_pagos no llovio, no feriado, fin de semana, centrico: {pax_pases_pagos_NPrecip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, fin de semana, residencial: {pax_pases_pagos_NPrecip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, dia de semana, centrico: {pax_pases_pagos_NPrecip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, no feriado, dia de semana, residencial: {pax_pases_pagos_NPrecip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, fin de semana, centrico: {pax_pases_pagos_NPrecip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, fin de semana, residencial: {pax_pases_pagos_NPrecip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, dia de semana, centrico: {pax_pases_pagos_NPrecip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos no llovio, feriado, dia de semana, residencial: {pax_pases_pagos_NPrecip_feri_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, fin de semana, centrico: {pax_pases_pagos_Precip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, fin de semana, residencial: {pax_pases_pagos_Precip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, dia de semana, centrico: {pax_pases_pagos_Precip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, no feriado, dia de semana, residencial: {pax_pases_pagos_Precip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, fin de semana, centrico: {pax_pases_pagos_Precip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, fin de semana, residencial: {pax_pases_pagos_Precip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, dia de semana, centrico: {pax_pases_pagos_Precip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_pases_pagos llovio, feriado, dia de semana, residencial: {pax_pases_pagos_Precip_feri_semana_Residencial / total_demanda * 100:.2f}%")

print('###########PAX_FREQ_PAGOS##############')

print(
    f"pax_freq no llovio, no feriado, fin de semana, centrico: {pax_freq_NPrecip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, fin de semana, residencial: {pax_freq_NPrecip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, dia de semana, centrico: {pax_freq_NPrecip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, no feriado, dia de semana, residencial: {pax_freq_NPrecip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, fin de semana, centrico: {pax_freq_NPrecip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, fin de semana, residencial: {pax_freq_NPrecip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, dia de semana, centrico: {pax_freq_NPrecip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq no llovio, feriado, dia de semana, residencial: {pax_freq_NPrecip_feri_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, fin de semana, centrico: {pax_freq_Precip_Nferi_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, fin de semana, residencial: {pax_freq_Precip_Nferi_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, dia de semana, centrico: {pax_freq_Precip_Nferi_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, no feriado, dia de semana, residencial: {pax_freq_Precip_Nferi_semana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, fin de semana, centrico: {pax_freq_Precip_feri_Nsemana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, fin de semana, residencial: {pax_freq_Precip_feri_Nsemana_Residencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, dia de semana, centrico: {pax_freq_Precip_feri_semana_NResidencial / total_demanda * 100:.2f}%")
print(
    f"pax_freq llovio, feriado, dia de semana, residencial: {pax_freq_Precip_feri_semana_Residencial / total_demanda * 100:.2f}%")

# # GRAFICOS
# ##Graficos por grupo de demanda
# # Datos actualizados para los tres grupos de demanda
# import matplotlib.pyplot as plt
#
# # Datos actualizados para los tres grupos de demanda
# categorias = [
#     "no llovio, no feriado, fin de semana",
#     "no llovio, no feriado, día de semana",
#     "no llovio, feriado, fin de semana",
#     "no llovio, feriado, día de semana",
#     "Con llovio, no feriado, fin de semana",
#     "Con llovio, no feriado, día de semana",
#     "Con llovio, feriado, fin de semana",
#     "Con llovio, feriado, día de semana"
# ]
#
# # Porcentajes para cada grupo de demanda
# porcentajes_pax_pagos = [2.72, 61.92, 0.13, 1.74, 0.87, 24.33, 0.05, 0.82]
# porcentajes_pax_pases_pagos = [0.19, 68.02, 0.01, 1.59, 0.06, 26.67, 0.00, 1.02]
# porcentajes_pax_freq_pagos = [2.72, 62.46, 0.15, 1.65, 0.94, 24.33, 0.05, 1.05]
#
# # Reajustando la configuración para ubicar los porcentajes fuera de los gráficos correctamente
# fig, ax = plt.subplots(1, 4, figsize=(30, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.5]})
#
# # Colores para los gráficos de torta
# colors = plt.get_cmap('tab20').colors
#
# # Función para colocar los porcentajes fuera del gráfico
# def autopct_generator(limit):
#     def inner_autopct(pct):
#         return '{:.1f}%'.format(pct) if pct > limit else ''
#
#     return inner_autopct
#
# # Configurando los gráficos de torta con porcentajes fuera
# for i, porcentajes in enumerate([porcentajes_pax_pagos, porcentajes_pax_pases_pagos, porcentajes_pax_freq_pagos]):
#     wedges, texts, autotexts = ax[i].pie(
#         porcentajes, autopct=autopct_generator(0), startangle=140, colors=colors,
#         pctdistance=1.2)  # pctdistance controla la distancia de los porcentajes al centro del gráfico
#     ax[i].set_title(['PAX_PAGOS', 'PAX_PASES_PAGOS', 'PAX_FREQ_PAGOS'][i])
#     ax[i].axis('equal')
#
#     # Ajuste para mejorar la visibilidad de los porcentajes
#     for autotext in autotexts:
#         autotext.set_color('black')
#         autotext.set_fontsize(9)
#
# # Eliminando el eje del cuarto panel y agregando la leyenda
# ax[3].axis('off')
# ax[3].legend(wedges, categorias, title="Categorías", loc="center", prop={'size': 9})
#
# plt.tight_layout()
# plt.show()
#
# # Ahora, los gráficos de barras para cada grupo, para una comparación detallada
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
#
# # PAX_PAGOS
# axes[0].barh(categorias, porcentajes_pax_pagos, color='lightblue')
# axes[0].set_title('PAX_PAGOS')
# axes[0].set_xlabel('Porcentaje')
#
# # PAX_PASES_PAGOS
# axes[1].barh(categorias, porcentajes_pax_pases_pagos, color='lightgreen')
# axes[1].set_title('PAX_PASES_PAGOS')
# axes[1].set_xlabel('Porcentaje')
#
# # PAX_FREQ_PAGOS
# axes[2].barh(categorias, porcentajes_pax_freq_pagos, color='salmon')
# axes[2].set_title('PAX_FREQ_PAGOS')
# axes[2].set_xlabel('Porcentaje')
#
# plt.tight_layout()
# plt.show()
#
# # GRAFICOS
# ##Graficos tomando en cuenta demanda total
# # Datos sumados para cada tipo de pasajero
# from matplotlib import pyplot as plt
#
# total_pax_pagos = sum([2.62, 59.44, 0.13, 1.67, 0.84, 23.36, 0.05, 0.79])
# total_pax_pases_pagos = sum([0.00, 0.32, 0.00, 0.01, 0.00, 0.13, 0.00, 0.00])
# total_pax_freq_pagos = sum([0.10, 2.20, 0.01, 0.06, 0.03, 0.86, 0.00, 0.04])
#
# # Datos para el gráfico de torta
# sizes = [total_pax_pagos, total_pax_pases_pagos, total_pax_freq_pagos]
# labels = ['Pax Pagos', 'Pax Pases Pagos', 'Pax Freq Pagos']
# colors = ['#ff9999','#66b3ff','#99ff99']
# explode = (0.1, 0, 0)  # solo "explotar" el primer trozo (i.e., 'Pax Pagos')
#
# # Utilizando una función lambda para aumentar el espacio para los porcentajes
# autopct = lambda pct: f'{pct:.1f}%' if pct > 0 else ''
#
# # Ajustando el parámetro 'explode' para mejorar la visualización
# explode = (0.1, 0.3, 0.2)  # Separando todas las secciones para mejor claridad
#
# # Creando el gráfico de torta nuevamente con los ajustes
# fig1, ax1 = plt.subplots()
# wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=autopct,
#                                    shadow=True, startangle=90)
#
# # Mejorando la visibilidad de los porcentajes moviéndolos más hacia afuera
# for autotext in autotexts:
#     autotext.set_color('black')
# plt.setp(autotexts, size=8, weight="bold")
# ax1.axis('equal')  # La relación aspecto igual asegura que la torta sea circular.
#
# plt.title('Distribución de la demanda por tipo de pasajero')
# plt.show()
