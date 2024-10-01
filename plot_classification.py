import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read the CSV file
df = pd.read_csv('chess_results.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Scatter plot
colors = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'yellow'}
labels = {'TP': 'True Positive', 'TN': 'True Negative', 'FP': 'False Positive', 'FN': 'False Negative'}

for result in ['TP', 'TN', 'FP', 'FN']:
    mask = df['Result'] == result
    ax1.scatter(df.loc[mask, 'CPL'], df.loc[mask, 'ELO'],
                c=colors[result], label=labels[result], s=100, alpha=0.7)

# Customize the scatter plot
ax1.set_xlabel('Centipawn Loss (CPL)')
ax1.set_ylabel('ELO Rating')
ax1.set_title('Chess Cheat Detection Results')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Add annotations for each point
for idx, row in df.iterrows():
    ax1.annotate(f"{row['Result']}", (row['CPL'], row['ELO']),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

# Confusion Matrix
cm = confusion_matrix(df['Actual'], df['Predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')
ax2.set_xticklabels(['Non-Cheat', 'Cheat'])
ax2.set_yticklabels(['Non-Cheat', 'Cheat'])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('chess_results_plot_with_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot with confusion matrix saved as chess_results_plot_with_confusion_matrix.png")