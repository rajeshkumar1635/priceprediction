import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('nyc_dataset.csv')

# Force 'PRICE' column to numeric and coerce errors
data['PRICE'] = pd.to_numeric(data['PRICE'], errors='coerce')

# Perform basic descriptive statistics on numeric columns
data_description = data.describe()

# Calculate the complete count for 'PRICE' including NaN values if any
price_count = len(data['PRICE'])
price_non_null_count = data['PRICE'].notnull().sum()

# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Print the count for 'PRICE', basic descriptive statistics, and missing values
print(f"Total count for 'PRICE': {price_count}")
print(f"Non-null count for 'PRICE': {price_non_null_count}")
print("\nDescriptive Statistics:\n", data_description)
print("\nMissing Values:\n", missing_values)

# Exclude non-numeric columns for correlation calculation
numeric_cols = data.select_dtypes(include=[np.number])

# Correlation matrix for numeric columns
correlation_matrix = numeric_cols.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Visualizing the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Detect and print outliers based on IQR for each feature
outliers = {}
for column in numeric_cols.columns:
    Q1 = numeric_cols[column].quantile(0.25)
    Q3 = numeric_cols[column].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = ((numeric_cols[column] < (Q1 - 1.5 * IQR)) | (numeric_cols[column] > (Q3 + 1.5 * IQR)))
    outliers[column] = numeric_cols[column][outlier_condition].count()

print("\nOutliers detected (using IQR method):\n", outliers)

# Plotting distributions and box plots for each numeric feature
for batch_start in range(0, len(numeric_cols.columns), 2):
    batch_end = batch_start + 2
    selected_columns = numeric_cols.columns[batch_start:batch_end]
    
    plt.figure(figsize=(15, 8))  # Adjust figure size as necessary

    for i, column in enumerate(selected_columns):
        # Histogram
        plt.subplot(2, 2, i * 2 + 1)
        sns.histplot(numeric_cols[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(2, 2, i * 2 + 2)
        sns.boxplot(x=numeric_cols[column].dropna())
        plt.title(f'Boxplot of {column}')

    plt.tight_layout()
    plt.show()