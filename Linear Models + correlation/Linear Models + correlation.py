import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('Superstore Sales Dataset.csv')

category_mapping = {
    'Furniture': 1,
    'Office Supplies': 2,
    'Technology': 3
}
df['Category_Num'] = df['Category'].map(category_mapping)


data = df[['Category_Num', 'Sales']]

X = data[['Category_Num']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"Coeficiente de determinacion R2: {r2:.4f}")

print(f"Intercepto: {model.intercept_:.2f}")
print(f"Coeficiente para Category_Num: {model.coef_[0]:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Category_Num', y='Sales', data=data, alpha=0.5)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title('Regresion Lineal: Ventas vs Categoría')
plt.xlabel('Categoria (1=Furniture, 2=Office Supplies, 3=Technology)')
plt.ylabel('Ventas ($)')
plt.xticks([1, 2, 3], ['Furniture', 'Office Supplies', 'Technology'])
plt.show()