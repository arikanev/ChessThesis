import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.inspection import PartialDependenceDisplay

# Load the datasets
df_full = pd.read_csv('chess_dataset.csv')
df_results = pd.read_csv('chess_results.csv')

# 1. Heatmap of Cheating Probability
plt.figure(figsize=(12, 8))
heatmap_data = df_full.pivot_table(values='Cheat', index='ELO', columns='CPL', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Probability of Cheating'})
plt.title('Heatmap of Cheating Probability by ELO and CPL')
plt.xlabel('Centipawn Loss (CPL)')
plt.ylabel('ELO Rating')
plt.savefig('cheating_probability_heatmap.png')
plt.close()

# 2. Violin Plot of CPL Distribution by ELO Range
df_full['ELO_Range'] = pd.cut(df_full['ELO'], bins=[0, 1500, 2000, 2500, 3000, 3500], labels=['0-1500', '1501-2000', '2001-2500', '2501-3000', '3001-3500'])
plt.figure(figsize=(12, 6))
sns.violinplot(x='ELO_Range', y='CPL', hue='Cheat', data=df_full, split=True)
plt.title('CPL Distribution by ELO Range and Cheat Status')
plt.xlabel('ELO Range')
plt.ylabel('Centipawn Loss (CPL)')
plt.savefig('cpl_distribution_by_elo_violin.png')
plt.close()

# 3. Precision-Recall Curve
X = df_full[['CPL', 'ELO']]
y = df_full['Cheat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_proba = dt.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.close()

# 4. Feature Importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Decision Tree Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('feature_importance.png')
plt.close()

# 5. Partial Dependence Plots
fig, ax = plt.subplots(figsize=(12, 5))
PartialDependenceDisplay.from_estimator(dt, X, ['CPL', 'ELO'], ax=ax)
plt.suptitle('Partial Dependence Plots for CPL and ELO')
plt.savefig('partial_dependence_plots.png')
plt.close()

# 6. Decision Boundary Visualization
def plot_decision_boundary(X, y, model, ax=None):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    if ax is None:
        ax = plt.gca()
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_xlabel('CPL')
    ax.set_ylabel('ELO')
    return ax

plt.figure(figsize=(10, 8))
ax = plot_decision_boundary(X.values, y.values, dt)
plt.title('Decision Tree Decision Boundary')
plt.savefig('decision_boundary.png')
plt.close()

print("All advanced visualizations have been saved as PNG files.")