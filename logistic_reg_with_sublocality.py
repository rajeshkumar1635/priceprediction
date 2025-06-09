import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('nyc_dataset.csv')

# Convert PRICE into a binary outcome based on the median price
price_threshold = data['PRICE'].median()
data['PRICE_BINARY'] = (data['PRICE'] > price_threshold).astype(int)

# One-hot encode 'SUBLOCALITY'
data = pd.get_dummies(data, columns=['SUBLOCALITY'])

# Select features and target
features = ['BEDS', 'BATH', 'PROPERTYSQFT'] + [col for col in data.columns if 'SUBLOCALITY' in col]
X = data[features].values
y = data['PRICE_BINARY'].values

# Ensure X is a NumPy array
X = np.array(X, dtype=np.float64)

# Normalize the features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation

# Avoid division by zero in case there is a standard deviation of zero
std[std == 0] = 1

X = (X - mean) / std

# Add intercept term
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the cost
def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        weights -= (learning_rate / m) * np.dot(X.T, predictions - y)
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Initialize parameters
weights = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000

# Run gradient descent
final_weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)

# Prediction function
def predict(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

# Predict labels on the dataset
y_pred = predict(X, final_weights)

# Functions to calculate confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

# Print the confusion matrix
tp, tn, fp, fn = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print("          Predicted: No  Predicted: Yes")
print(f"Actual: No  TN: {tn}      FP: {fp}")
print(f"Actual: Yes FN: {fn}      TP: {tp}")

# Functions to calculate precision, recall, F1 score, and accuracy
def precision_recall_f1_accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy

# Calculate metrics
precision, recall, f1, accuracy = precision_recall_f1_accuracy(y, y_pred)

# Output the results
print("Final weights:", final_weights)
print("Final cost:", cost_history[-1])
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)
