import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Superstore Sales Dataset.csv')

category_mapping = {
    'Furniture': 0,
    'Office Supplies': 1,
    'Technology': 2
}

data['Category_Num'] = data['Category'].map(category_mapping)

features = data[['Sales', 'Category_Num']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Metodo del Codo para Determinar k Óptimo')
plt.xlabel('Numero de clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_features)

data['Cluster'] = clusters

print("Distribucion de clusters:")
print(data['Cluster'].value_counts())

print("\nDistribucion de categorias por cluster:")
print(pd.crosstab(data['Cluster'], data['Category']))

plt.figure(figsize=(12, 7))
sns.scatterplot(x=data['Sales'], y=data['Category_Num'], hue=data['Cluster'], 
                palette='viridis', style=data['Category'], s=100)
plt.title('Clusters de Ventas por Categoria de Producto')
plt.xlabel('Ventas (Sales)')
plt.ylabel('Categoría (0=Furniture, 1=Office Supplies, 2=Technology)')
plt.yticks([0, 1, 2], ['Furniture', 'Office Supplies', 'Technology'])
plt.show()

print("\nAnalisis estadistico por cluster:")
cluster_stats = data.groupby('Cluster').agg({
    'Sales': ['mean', 'median', 'min', 'max', 'count'],
    'Category_Num': 'mean'
})
print(cluster_stats)