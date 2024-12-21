import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Download latest version
path = kagglehub.dataset_download("ruchi798/movies-on-netflix-prime-video-hulu-and-disney")

df_ori = pd.read_csv(path+"\MoviesOnStreamingPlatforms.csv")

# Set weight of age restriction
age_weights = {
    'all': 0,
    '7+': 0.1,
    '13+': 0.3,
    '16+': 0.6,
    '18+': 1
}

# Map age weight to films
df_ori['Age Weight'] = df_ori['Age'].map(age_weights)

# Score Conversion
df_ori['RT Score'] = df_ori['Rotten Tomatoes'].dropna().str.split('/').str[0].astype(int)

# Split Disney+ films and Netflix films
df_Disneyp = df_ori.loc[df_ori['Disney+'] == 1]
df_Netflix = df_ori.loc[df_ori['Netflix'] == 1]

# Clear missing age restriction values
df_Disneyp_age_cleaned = df_Disneyp.dropna(subset=['Age'])
df_Netflix_age_cleaned = df_Netflix.dropna(subset=['Age'])

print(df_Disneyp_age_cleaned['Age Weight'].mean())
print(df_Netflix_age_cleaned['Age Weight'].mean())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
color_dict = {'all': '#0000ff', '7+': '#00ff00', '13+': '#ffff00', '16+': '#ffa500', '18+':'#ff0000'}

# Print pie-plot
df_Disneyp_age_cnts = df_Disneyp_age_cleaned['Age'].value_counts()
df_Disneyp_age_cnts.plot(kind='pie', autopct='%1.1f%%', ax=axes[0], colors=[color_dict[age] for age in df_Disneyp_age_cnts.index])
axes[0].set_title('Disney+ Films Age Restriction Rate')
df_Netflix_age_cnts = df_Netflix_age_cleaned['Age'].value_counts()
df_Netflix_age_cnts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1], colors=[color_dict[age] for age in df_Netflix_age_cnts.index])
axes[1].set_title('Netflix Films Age Restriction Rate')

plt.tight_layout() 
plt.show()

df_Disneyp_RT_cleaned = df_Disneyp.dropna(subset=['Rotten Tomatoes'])
df_Netflix_RT_cleaned = df_Netflix.dropna(subset=['Rotten Tomatoes'])

print(df_Disneyp_RT_cleaned['RT Score'].describe())
print(df_Netflix_RT_cleaned['RT Score'].describe())

df_score = pd.DataFrame({
    'RT Score': pd.concat([df_Disneyp_RT_cleaned['RT Score'], df_Netflix_RT_cleaned['RT Score']]),
    'Platform': ['Disney+'] * len(df_Disneyp_RT_cleaned) + ['Netflix'] * len(df_Netflix_RT_cleaned)
})

df_score.boxplot(by= 'Platform', column= 'RT Score')
plt.show()

print(stats.mannwhitneyu(df_Disneyp_RT_cleaned['RT Score'], df_Netflix_RT_cleaned['RT Score']))
