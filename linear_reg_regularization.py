import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
file_path = 'nyc_dataset.csv'
data = pd.read_csv(file_path)
relevant_columns = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'SUBLOCALITY']
data = data[relevant_columns]

# Normalize numeric features including dummy variables
numeric_features = ['BEDS', 'BATH', 'PROPERTYSQFT']
for feature in numeric_features:
    mean = data[feature].mean()
    std = data[feature].std()
    data[feature] = (data[feature] - mean) / std

# Manual One-hot encoding for 'SUBLOCALITY' and standardize
sublocalities = data['SUBLOCALITY'].unique()
for subloc in sublocalities:
    data[f'subloc_{subloc}'] = (data['SUBLOCALITY'] == subloc).astype(int)
data.drop('SUBLOCALITY', axis=1, inplace=True)

# Standardize all features (including dummy variables)
for feature in data.columns:
    if feature != 'PRICE':
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = (data[feature] - mean) / std

# Separate target variable
y = data['PRICE'].values
data.drop('PRICE', axis=1, inplace=True)

# Forward Stepwise Regression with Regularization
def compute_cost(X, y, theta, lambda_):
    predictions = X @ theta
    errors = predictions - y
    regularization_term = lambda_ * np.sum(np.square(theta[1:]))  # exclude intercept
    return (1 / (2 * len(y))) * (np.dot(errors.T, errors) + regularization_term)

lambda_ = 1  # Regularization parameter
features = list(data.columns)
selected_features = []
cost_history = []

while features:
    costs = []
    for feature in features:
        current_features = selected_features + [feature]
        X = data[current_features].values
        X = np.c_[np.ones(len(X)), X]  # Add intercept term
        # Regularized normal equation
        XTX = X.T @ X
        lambda_I = lambda_ * np.eye(XTX.shape[0])
        lambda_I[0, 0] = 0  # Do not regularize intercept
        theta = np.linalg.inv(XTX + lambda_I) @ X.T @ y
        cost = compute_cost(X, y, theta, lambda_)
        costs.append((cost, feature))
    
    costs.sort()
    min_cost, best_feature = costs[0]
    if not selected_features or min_cost < cost_history[-1]:
        selected_features.append(best_feature)
        features.remove(best_feature)
        cost_history.append(min_cost)
        print(f"Selected {best_feature}, Cost: {min_cost}")
    else:
        print(f"Stopped adding features after {best_feature} due to no improvement.")
        break

# Final model with selected features
X_final = data[selected_features].values
X_final = np.c_[np.ones(len(X_final)), X_final]
theta_final = np.linalg.inv(X_final.T @ X_final + lambda_ * np.eye(X_final.shape[1])) @ X_final.T @ y
final_predictions = X_final @ theta_final
final_cost = compute_cost(X_final, y, theta_final, lambda_)

# Calculate R-squared and Adjusted R-squared
rss = np.sum((y - final_predictions) ** 2)
tss = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (rss / tss)
adjusted_r_squared = 1 - ((1 - r_squared) * (len(y) - 1)) / (len(y) - len(selected_features) - 1)
mse = rss / len(y)

print(f"Final Model with Regularization: R-squared: {r_squared}, Adjusted R-squared: {adjusted_r_squared}, MSE: {mse}")
print("Selected Features with Regularization:", selected_features)

# Plot the cost function over iterations
plt.plot(cost_history, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cost Function Value')
plt.title('Cost Function History with Regularization')
plt.show()
