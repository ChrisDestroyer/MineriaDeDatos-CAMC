import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv('Superstore Sales Dataset.csv')

product_names = data['Product Name'].dropna()

text = ' '.join(product_names)

word_freq = Counter(text.split())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Product Names in Superstore Sales Dataset')
plt.show()