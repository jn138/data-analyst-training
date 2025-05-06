import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
original_filename = "train_subset.csv"
df = pd.read_csv(original_filename)

# Set up the figure with 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Scatter Plot 1: Height vs. Weight
axes[0].scatter(df['height'], df['weight'], alpha=0.6, color='teal')
axes[0].set_title('height vs. weight')
axes[0].set_xlabel('Height (cm)')
axes[0].set_ylabel('Weight (kg)')
axes[0].grid(True)

# Scatter Plot 2: Systolic vs Diastolic Blood Pressure
axes[1].scatter(df['ap_hi'], df['ap_lo'], alpha=0.6, color='purple')
axes[1].set_title('ap_hi vs. ap_lo')
axes[1].set_xlabel('Systolic (ap_hi)')
axes[1].set_ylabel('Diastolic (ap_lo)')
axes[1].grid(True)

# Adjust layout for readability
plt.tight_layout()
plt.show()
