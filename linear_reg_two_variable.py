import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'nyc_dataset.csv'
data = pd.read_csv(file_path)

# Select features and target variable
X = data[['BEDS', 'BATH', 'PROPERTYSQFT']].values
y = data['PRICE'].values

# Randomly shuffle the dataset
np.random.seed(0)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.8)

X_train = X[indices[:split_idx]]
y_train = y[indices[:split_idx]]
X_test = X[indices[split_idx:]]
y_test = y[indices[split_idx:]]

# Standardize the training and test features
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_standardized = (X_train - X_train_mean) / X_train_std
X_test_standardized = (X_test - X_train_mean) / X_train_std

# Add intercept term to both training and test sets
X_train_b = np.c_[np.ones(X_train_standardized.shape[0]), X_train_standardized]
X_test_b = np.c_[np.ones(X_test_standardized.shape[0]), X_test_standardized]

# Gradient Descent and Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)
    
    for i in range(num_iterations):
        predictions = X.dot(theta)
        theta -= (1/m) * learning_rate * (X.T.dot(predictions - y))
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# Initialize parameters
theta = np.zeros(X_train_b.shape[1])
learning_rate = 0.01
num_iterations = 500

# Run gradient descent on training data
theta, cost_history = gradient_descent(X_train_b, y_train, theta, learning_rate, num_iterations)

# Predictions on test data
y_test_pred = X_test_b.dot(theta)

# Compute MSE for test data
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_test = mean_squared_error(y_test, y_test_pred)

# Compute R-squared and Adjusted R-squared
def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def adjusted_r_squared(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

r2_test = r_squared(y_test, y_test_pred)
adjusted_r2_test = adjusted_r_squared(r2_test, len(y_test), 3)

# Output results
print("Thetas: ", theta)
print("MSE for test data: ", mse_test)
print("r2 for test data: ", r2_test)
print("adjusted r2 for test data: ", adjusted_r2_test)

# Plot the cost history from training
plt.figure(figsize=(12, 6))
plt.plot(cost_history, 'b-')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function Value')
plt.title('Cost Function History during Training')
plt.grid(True)
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices on Test Data')
plt.legend()
plt.grid(True)
plt.show()
