import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('chess_cpl_data.csv')

# Filter the data for ELO 1001
df_1001 = df[df['ELO'] == 1003]

# Create the plot
plt.figure(figsize=(12, 6))
sns.histplot(data=df_1001, x='CPL', hue='Is_Cheating', kde=True, element='bars', stat='density', common_norm=False)

# Customize the plot
plt.title('CPL Distribution for Cheating vs Non-Cheating Players (ELO 1002)', fontsize=16)
plt.xlabel('Centipawn Loss (CPL)', fontsize=12)
plt.ylabel('Density', fontsize=12)

# Add a legend with clear labels
plt.legend(title='Is Cheating', title_fontsize='12', fontsize='10', labels=['No', 'Yes'])

# Improve the layout
plt.tight_layout()

# Save the plot
plt.savefig('cpl_distribution_elo_1002_corrected.png', dpi=300)
plt.close()

print("The corrected CPL distribution plot for ELO 1001 has been saved as 'cpl_distribution_elo_1001_corrected.png'.")

# Print some summary statistics
print("\nSummary Statistics for ELO 1002:")
print(df_1001.groupby('Is_Cheating')['CPL'].describe())

# Calculate and print the percentage of cheating players
cheating_percentage = (df_1001['Is_Cheating'].sum() / len(df_1001)) * 100
print(f"\nPercentage of cheating players at ELO 1001: {cheating_percentage:.2f}%")