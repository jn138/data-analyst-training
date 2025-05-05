# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Declare file names
training_filename = "train_subset.csv"
testing_filename = "test_kaggle_features.csv"
output_filename = "test_kaggle_predictions.csv"

# Load the datasets
d1 = pd.read_csv(training_filename)
d2 = pd.read_csv(testing_filename)

# Convert "Yes" to 1 and "No" to 0 within certain columns
columns = ['smoke', 'alco', 'active', 'cardio']
for column in columns:
  d1[column] = d1[column].apply(lambda x: 1 if x == 'Yes' else 0)

  # Same for testing data except 'cardio' (not in the dataset)
  if column != 'cardio':
    d2[column] = d2[column].apply(lambda x: 1 if x == 'Yes' else 0)

# Prepare features and labels
X_train = d1.drop(['id', 'cardio'], axis=1)
Y_train = d1['cardio']
X_test = d2.drop(['id'], axis=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the prediction model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_predict = model.predict(X_test_scaled)

# Convert back 0 to "No" and 1 to "Yes"
Y_predict = ['Yes' if pred == 1 else 'No' for pred in Y_predict]

# Export output to CSV
output = pd.DataFrame({
  'id': d2['id'],
  'cardio': Y_predict
})
output.to_csv(output_filename, index=False)