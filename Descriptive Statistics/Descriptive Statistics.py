import pandas as pd

df = pd.read_csv('Superstore Sales Dataset.csv')

#Análisis Descriptivo General
print("\nEstadisticas Descriptivas Generales")
print(df.describe(include='all'))


#Ventas por Categoría de Producto
sales_by_category = df.groupby('Category')['Sales'].agg(['sum', 'mean', 'count'])
print("\nVentas por Categoria")
print(sales_by_category)

#Ventas por Región
sales_by_region = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count'])
print("\nVentas por Region")
print(sales_by_region)

#Ventas por Segmento de Cliente
sales_by_segment = df.groupby('Segment')['Sales'].agg(['sum', 'mean', 'count'])
print("\nVentas por Segmento de Cliente")
print(sales_by_segment)

#Ventas por Modo de Envío
sales_by_shipmode = df.groupby('Ship Mode')['Sales'].agg(['sum', 'mean', 'count'])
print("\nVentas por Modo de Envio")
print(sales_by_shipmode)
