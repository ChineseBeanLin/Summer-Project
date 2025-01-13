import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.contingency_tables as sm
import numpy as np
from scipy import stats

# Download latest version
path = kagglehub.dataset_download("ruchi798/movies-on-netflix-prime-video-hulu-and-disney")

df_ori = pd.read_csv(path+"\MoviesOnStreamingPlatforms.csv")

# Set order of age restriction
age_order = ['all', '7+', '13+', '16+', '18+']
focus_keys = ['ID', 'Netflix', 'Disney+', 'Age', 'RT Score']

# Score Conversion
df_ori['RT Score'] = df_ori['Rotten Tomatoes'].dropna().str.split('/').str[0].astype(int)

# show both in Disney+ and Netflix
print(df_ori[(df_ori['Disney+'] == 1) & (df_ori['Netflix'] == 1)])

# Split Disney+ films and Netflix films
df_Disneyp = df_ori.loc[df_ori['Disney+'] == 1]
df_Netflix = df_ori.loc[df_ori['Netflix'] == 1]

# count the number of films in platforms
print('Number of films in Disney+:', len(df_Disneyp))
print('Number of films in Netflix:', len(df_Netflix))

# Clear missing age restriction values
df_Disneyp_age_cleaned = df_Disneyp.dropna(subset=['Age'])
df_Netflix_age_cleaned = df_Netflix.dropna(subset=['Age'])

missing_data_Disneyp = df_Disneyp[df_Disneyp['Age'].isna()]
missing_data_Netflix = df_Netflix[df_Netflix['Age'].isna()]
print(missing_data_Disneyp[focus_keys].head())
print(missing_data_Netflix[focus_keys].head())

# Check the distribution of age restriction
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
color_dict = {'all': '#cedfef', '7+': '#92c2dd', '13+': '#4995c6', '16+': '#1663a9', '18+':'#614099', 'other':'#a0a0a0'}

# Set age restriction as ordered categorical data
df_Disneyp_age_cleaned.loc[:, 'Age'] = pd.Categorical(df_Disneyp_age_cleaned['Age'], categories=age_order, ordered=True)
df_Netflix_age_cleaned.loc[:, 'Age'] = pd.Categorical(df_Netflix_age_cleaned['Age'], categories=age_order, ordered=True)

df_Disneyp_age_cnts = df_Disneyp_age_cleaned['Age'].value_counts().reindex(age_order)
df_Netflix_age_cnts = df_Netflix_age_cleaned['Age'].value_counts().reindex(age_order)

# Calculate the proportion of each age restriction
print('Disney+:'+ str((df_Disneyp_age_cnts / df_Disneyp_age_cnts.sum()).round(4)))
print('Netflix:'+ str((df_Netflix_age_cnts / df_Netflix_age_cnts.sum()).round(4)))
print(df_Disneyp_age_cnts)
print(df_Netflix_age_cnts)

# Combine age restriction with less than 50 films
df_Disneyp_age_cnts_draw = df_Disneyp_age_cnts.copy()
small_index = df_Disneyp_age_cnts_draw[df_Disneyp_age_cnts_draw < 50].index
df_Disneyp_age_cnts_draw['other'] = df_Disneyp_age_cnts_draw[small_index].sum()
df_Disneyp_age_cnts_draw = df_Disneyp_age_cnts_draw.drop(small_index)
print(df_Disneyp_age_cnts_draw)

# Plot pie-plot
df_Disneyp_age_cnts_draw.plot(startangle=30, kind='pie', autopct='%1.1f%%', ax=axes[0], colors=[color_dict[age] for age in df_Disneyp_age_cnts_draw.index])
axes[0].set_title('Disney+ Films Age Restriction Rate')
df_Netflix_age_cnts.plot(startangle=30, kind='pie', autopct='%1.1f%%', ax=axes[1], colors=[color_dict[age] for age in df_Netflix_age_cnts.index])
axes[1].set_title('Netflix Films Age Restriction Rate')

plt.tight_layout()
plt.show()

# Check the distribution of Rotten Tomatoes score for trend
# Create a contingency table
age_platform_table = pd.DataFrame({
    'Disney+': df_Disneyp_age_cnts,
    'Netflix': df_Netflix_age_cnts
})

print(age_platform_table)
# use test_ordinal_association default score(Cochran–Armitage test for trend)
table = sm.Table(age_platform_table)
result = table.test_ordinal_association()
print(result)

# Analyse Rotten Tomatoes Scores
# Clear Missing Rotten Tomatoes Scores Values
df_Disneyp_RT_cleaned = df_Disneyp.dropna(subset=['Rotten Tomatoes'])
df_Netflix_RT_cleaned = df_Netflix.dropna(subset=['Rotten Tomatoes'])
print(df_Disneyp[df_Disneyp['Rotten Tomatoes'].isna()][focus_keys])
print(df_Netflix[df_Netflix['Rotten Tomatoes'].isna()][focus_keys])

# Check descriptive statistics
print(', '.join(df_Disneyp_RT_cleaned['RT Score'].describe().round(2).astype(str).values))
print(', '.join(df_Netflix_RT_cleaned['RT Score'].describe().round(2).astype(str).values))

# Create Boxplot
df_score = pd.DataFrame({
    'RT Score': pd.concat([df_Disneyp_RT_cleaned['RT Score'], df_Netflix_RT_cleaned['RT Score']]),
    'Platform': ['Disney+'] * len(df_Disneyp_RT_cleaned) + ['Netflix'] * len(df_Netflix_RT_cleaned)
})

df_score.boxplot(by= 'Platform', column= 'RT Score')
plt.show()

# check normality
# use Shapiro-Wilk test to check normality
print(stats.shapiro(df_Disneyp_RT_cleaned['RT Score']))
print(stats.shapiro(df_Netflix_RT_cleaned['RT Score']))

# use Q-Q plot to check normality
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
stats.probplot(df_Disneyp_RT_cleaned['RT Score'], plot=axes[0])
axes[0].set_title('Disney+ RT Score Q-Q Plot')
stats.probplot(df_Netflix_RT_cleaned['RT Score'], plot=axes[1])
axes[1].set_title('Netflix RT Score Q-Q Plot')
plt.show()

# check difference between Disney+ and Netflix RT Score
# use Mann-Whitney U test to compare the RT Score of Disney+ and Netflix
print(stats.mannwhitneyu(df_Disneyp_RT_cleaned['RT Score'], df_Netflix_RT_cleaned['RT Score']))