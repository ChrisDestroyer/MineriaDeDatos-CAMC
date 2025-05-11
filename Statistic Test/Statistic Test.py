import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv('Superstore Sales Dataset.csv')

f_stat, p_val = stats.f_oneway(
    df[df['Category'] == 'Furniture']['Sales'],
    df[df['Category'] == 'Office Supplies']['Sales'],
    df[df['Category'] == 'Technology']['Sales']
)
print(f"ANOVA: F={f_stat:.3f}, p={p_val:.4f}")

if p_val < 0.05:
    tukey = pairwise_tukeyhsd(df['Sales'], df['Category'])
    print(tukey.summary())

h_stat, p_val = stats.kruskal(
    df[df['Region'] == 'Central']['Sales'],
    df[df['Region'] == 'East']['Sales'],
    df[df['Region'] == 'South']['Sales'],
    df[df['Region'] == 'West']['Sales']
)
print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.4f}")