import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Superstore Sales Dataset.csv')

data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')

data['Year'] = data['Order Date'].dt.year

annual_sales = data.groupby('Year')['Sales'].sum().reset_index()

X = annual_sales[['Year']].values
y = annual_sales['Sales'].values

model = LinearRegression()
model.fit(X, y)

future_years = np.array([[2019], [2020], [2021], [2022], [2023]])
predictions = model.predict(future_years)

future_sales = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Sales': predictions
})

plt.figure(figsize=(8, 5))
plt.plot(annual_sales['Year'], annual_sales['Sales'], marker='o', linestyle='-', label='Ventas Reales')
plt.plot(future_sales['Year'], future_sales['Predicted Sales'], marker='x', linestyle='--', color='red', label='Predicciones')
plt.title('Prediccion de Ventas por Año')
plt.xlabel('Año')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True)
plt.show()

print(future_sales)