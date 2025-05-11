import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Superstore Sales Dataset.csv')

plt.style.use('ggplot')

#Graficos por categoria usando for loop
categories = df['Category'].unique()
plt.figure(figsize=(15, 10))
for i, category in enumerate(categories, 1):
    plt.subplot(2, 2, i)
    df[df['Category'] == category].groupby('Sub-Category')['Sales'].sum().plot.bar()
    plt.title(f'Ventas por Sub-Categoria: {category}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Graficos de caja por region usando for loop
regions = df['Region'].unique()
plt.figure(figsize=(15, 5))
for i, region in enumerate(regions, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y='Sales', data=df[df['Region'] == region])
    plt.title(f'Distribucion Ventas: {region}')
    plt.yscale('log')
plt.tight_layout()
plt.show()


#Grafico de pastel ventas por segmento
plt.figure(figsize=(15, 5))
ventas_por_segmento = df.groupby('Segment')['Sales'].sum()
plt.pie(ventas_por_segmento, 
        labels=ventas_por_segmento.index, 
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05, 0.05),
        shadow=True,
        colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Distribucion de Ventas por Segmento (%)', fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()