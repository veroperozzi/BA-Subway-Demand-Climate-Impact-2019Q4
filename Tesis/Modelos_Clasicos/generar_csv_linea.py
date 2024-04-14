import csv
from datetime import datetime

# Cambiar nombre CSV de entrada y salida
input_filename = "molinetes_2016_2017_2018_2019_concatenados_V1.0.csv"
output_filename = "linea_A_col_total_mensual.csv"

# Filtro a usar en LINEA
LINEA = "A"

# Abrir CSV y leer CSV
with open(input_filename) as input_file:
    reader = csv.reader(input_file)
    header = next(reader)  # Leer fila de encabezado

    # Crear diccionario para almacenar los totales mensuales
    monthly_totals = {}

    # Iterate over the rows in the input CSV file
    for row in reader:
        # If the LINEA column is equal to the desired value, get the month and year from the FECHA column and add the TOTAL column value to the corresponding entry in the monthly_totals dictionary
        if row[1] == LINEA:
            fecha = datetime.strptime(row[0], "%Y-%m-%d")  # Convert the FECHA column value to a datetime object
            month_year = fecha.strftime("%Y-%m")  # Get the month and year in the format "yyyy-mm"
            total = int(float(row[5]))  # Get the TOTAL column value as an integer

            if month_year in monthly_totals:
                monthly_totals[month_year] += total
            else:
                monthly_totals[month_year] = total

    # Open the output CSV file and create a CSV writer
    with open(output_filename, "w", newline="") as output_file:
        writer = csv.writer(output_file)

        # Write the header row to the output CSV file
        writer.writerow(["FECHA", "TOTAL"])

        # Iterate over the monthly_totals dictionary and write the month and total to the output CSV file
        for month_year, total in monthly_totals.items():
            # Convert the month and year back to a datetime object with day set to 1
            fecha = datetime.strptime(month_year + "-01", "%Y-%m-%d")

            # Convert the datetime object back to a string in the format "yyyy-mm-dd"
            fecha_str = fecha.strftime("%Y-%m-%d")

            writer.writerow([fecha_str, total])

