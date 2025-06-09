import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'nyc_dataset.csv'
data = pd.read_csv(file_path)
relevant_columns = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'SUBLOCALITY']
data = data[relevant_columns]

# Normalize numeric features
numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
for feature in numeric_features:
    mean = data[feature].mean()
    std = data[feature].std()
    data[feature] = (data[feature] - mean) / std

# Manual One-hot encoding for 'SUBLOCALITY'
sublocalities = data['SUBLOCALITY'].unique()
for subloc in sublocalities:
    data[f'subloc_{subloc}'] = (data['SUBLOCALITY'] == subloc).astype(int)
data.drop('SUBLOCALITY', axis=1, inplace=True)

# Separate target variable
y = data['PRICE'].values
data.drop('PRICE', axis=1, inplace=True)

# Forward Stepwise Regression
def compute_cost(X, y, theta):
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * len(y))) * np.dot(errors.T, errors)

features = list(data.columns)
selected_features = []
cost_history = []

while features:
    costs = []
    for feature in features:
        current_features = selected_features + [feature]
        X = data[current_features].values
        X = np.c_[np.ones(len(X)), X]  # Add intercept term
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        cost = compute_cost(X, y, theta)
        costs.append((cost, feature))
    
    # Sort and select the best feature based on cost
    costs.sort()
    min_cost, best_feature = costs[0]
    
    # Check if adding the new feature reduces the overall cost
    if not selected_features or min_cost < cost_history[-1]:
        selected_features.append(best_feature)
        features.remove(best_feature)
        cost_history.append(min_cost)
        print(f"Selected {best_feature}, Cost: {min_cost}")
    else:
        for cost, feature in costs:
            if feature not in selected_features:
                print(f"Not selected {feature}, Cost: {cost} - No improvement")
        print(f"Stopped adding features after {best_feature} due to no improvement.")
        break

# Final model with selected features
X_final = data[selected_features].values
X_final = np.c_[np.ones(len(X_final)), X_final]
theta_final = np.linalg.inv(X_final.T @ X_final) @ X_final.T @ y
final_predictions = X_final @ theta_final
final_cost = compute_cost(X_final, y, theta_final)

# Calculate R-squared and Adjusted R-squared
rss = np.sum((y - final_predictions) ** 2)
tss = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (rss / tss)
adjusted_r_squared = 1 - ((1 - r_squared) * (len(y) - 1)) / (len(y) - len(selected_features) - 1)
mse = rss / len(y)

print(f"Final Model: R-squared: {r_squared}, Adjusted R-squared: {adjusted_r_squared}, MSE: {mse}")
print("Selected Features:", selected_features)

# Plot the cost function over iterations
plt.plot(cost_history, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cost Function Value')
plt.title('Cost Function History during Feature Selection')
plt.show()
