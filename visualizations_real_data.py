import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('chess_cpl_data.csv')

# 1. Distribution of CPL for cheaters and non-cheaters
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='CPL', hue='Is_Cheating', kde=True, element='step')
plt.title('Distribution of CPL for Cheaters and Non-Cheaters')
plt.xlabel('Centipawn Loss (CPL)')
plt.ylabel('Count')
plt.savefig('cpl_distribution.png')
plt.close()

# 2. ELO Rating vs Cheating Probability
X = df[['CPL', 'ELO']]
y = df['Is_Cheating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['ELO'], df['CPL'], c=dt.predict_proba(df[['CPL', 'ELO']])[:, 1], cmap='coolwarm')
plt.colorbar(scatter, label='Predicted Probability of Cheating')
plt.title('ELO Rating vs CPL, Colored by Predicted Probability of Cheating')
plt.xlabel('ELO Rating')
plt.ylabel('Centipawn Loss (CPL)')
plt.savefig('elo_cpl_cheat_probability.png')
plt.close()

# 3. Model Performance across ELO Ranges
df['ELO_Range'] = pd.cut(df['ELO'], bins=[0, 1500, 2000, 2500, 3000, 3500], labels=['0-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500'])
df['Predicted'] = dt.predict(df[['CPL', 'ELO']])
accuracy_by_elo = df.groupby('ELO_Range').apply(lambda x: (x['Is_Cheating'] == x['Predicted']).mean())

plt.figure(figsize=(10, 6))
accuracy_by_elo.plot(kind='bar')
plt.title('Model Accuracy Across ELO Ranges')
plt.xlabel('ELO Range')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracy_by_elo):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.savefig('accuracy_by_elo.png')
plt.close()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(df['Is_Cheating'], dt.predict_proba(df[['CPL', 'ELO']])[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

print("All visualizations have been saved as PNG files.")