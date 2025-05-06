import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
original_filename = "train_subset.csv"
df = pd.read_csv(original_filename)

# # Set up the figure with 1 row, 2 columns
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Scatter Plot 1: Height vs. Weight
# axes[0].scatter(df['height'], df['weight'], alpha=0.6, color='teal')
# axes[0].set_title('height vs. weight')
# axes[0].set_xlabel('Height (cm)')
# axes[0].set_ylabel('Weight (kg)')
# axes[0].grid(True)

# # Scatter Plot 2: Systolic vs Diastolic Blood Pressure
# axes[1].scatter(df['ap_hi'], df['ap_lo'], alpha=0.6, color='purple')
# axes[1].set_title('ap_hi vs. ap_lo')
# axes[1].set_xlabel('Systolic (ap_hi)')
# axes[1].set_ylabel('Diastolic (ap_lo)')
# axes[1].grid(True)

# # Adjust layout for readability
# plt.tight_layout()
# plt.show()

# IQR method to filter out outliers
def remove_outliers_iqr(data, column):
  Q1 = data[column].quantile(0.25)
  Q3 = data[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Print number of records in the original dataset
print("Original records:", df.shape[0])

# Remove the outliers from the specified attributes
cleansing_attr = ["height", "weight", "ap_hi", "ap_lo"]
for attr in cleansing_attr:
  df = remove_outliers_iqr(df, attr)

# Print number of records in the cleansed dataset
print("Cleansed records:", df.shape[0])

# Save cleaned data to a new CSV file
df.to_csv("train_subset_cleaned.csv", index=False)
