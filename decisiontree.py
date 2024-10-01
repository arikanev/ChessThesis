import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
df = pd.read_csv('chess_dataset.csv')

# Prepare the features (X) and labels (y)
X = df[['CPL', 'ELO']].values
y = df['Cheat'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = decision_tree.predict(X_test)

# Create a results dataframe for the test set
results_df = pd.DataFrame({
    'CPL': X_test[:, 0],
    'ELO': X_test[:, 1],
    'Actual': y_test,
    'Predicted': y_test_pred
})

# Add a column for classification result
results_df['Result'] = np.where(results_df['Actual'] == results_df['Predicted'],
                                np.where(results_df['Actual'] == 1, 'TP', 'TN'),
                                np.where(results_df['Actual'] == 1, 'FN', 'FP'))

# Save the results to a CSV file
results_df.to_csv('chess_results.csv', index=False)

# Print classification report and confusion matrix for the test set
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# Optional: Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(decision_tree, filled=True, feature_names=['CPL', 'ELO'], class_names=['Non-Cheat', 'Cheat'], rounded=True)
plt.savefig('decision_tree_plot.png')
plt.close()

print("Test set results saved to chess_results.csv")
print("Decision tree plot saved to decision_tree_plot.png")