# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the training dataset
training_filename = "train_subset_cleaned.csv"
d1 = pd.read_csv(training_filename)


# Convert "Yes" to 1 and "No" to 0 within certain columns
columns = ['smoke', 'alco', 'active', 'cardio']
for column in columns:
  d1[column] = d1[column].apply(lambda x: 1 if x == 'Yes' else 0)

# Prepare features and labels
X = d1.drop(['id', 'cardio'], axis=1)
Y = d1['cardio']

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Find best parameters to build the model
param_grid = {
  'n_estimators': [200, 300, 400],    # Number of trees
  'max_depth': [10, 20, None],        # How deep the trees go
  'min_samples_split': [2, 3, 4],     # Min samples to split a node
  'min_samples_leaf': [2, 3, 4],      # Min samples at leaf node
  'class_weight': ['balanced']        # Tackle class imbalance
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',                # Optimizing for AUC score
    cv=5,                             # 5-fold cross-validation
    n_jobs=-1,                        # Use all cores
    verbose=1
)
grid_search.fit(X_train_scaled, Y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train the prediction model
# model = RandomForestClassifier(random_state=42)
model = best_model
model.fit(X_train_scaled, Y_train)

# Get predicted probabilities for the positive class (cardio = 1)
Y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Print the AUC score
auc = roc_auc_score(Y_test, Y_prob)
print(f"AUC Score: {auc:.4f}")

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()