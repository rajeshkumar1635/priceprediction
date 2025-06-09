import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('nyc_dataset.csv')

# Assume 'PRICE' is the feature and we're predicting a binary target ('High' price: 1, 'Low' price: 0)
data['Target'] = (data['PRICE'] > data['PRICE'].median()).astype(int)
features = data[['BEDS', 'BATH', 'PROPERTYSQFT']]
target = data['Target']

# Normalize features
features = (features - features.mean()) / features.std()

# Adding intercept term
features['Intercept'] = 1

# Split data into training and testing
np.random.seed(42)
train_indices = np.random.rand(len(features)) < 0.8
X_train = features[train_indices].values
X_test = features[~train_indices].values
y_train = target[train_indices].values
y_test = target[~train_indices].values

# Sigmoid function with clipping
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip values to avoid extremes
    return 1 / (1 + np.exp(-z))

# Loss function with epsilon to avoid log(0)
def compute_loss(y, h):
    epsilon = 1e-15  # Small value to ensure numerical stability
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

# Gradient Descent
def gradient_descent(X, y, num_iterations, learning_rate):
    weights = np.zeros(X.shape[1])
    loss_history = []
    for i in range(num_iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        weights -= learning_rate * gradient
        
        # Record the loss
        loss = compute_loss(y, h)
        loss_history.append(loss)
        
        # Print the loss every 1000 iterations
        if i % 1000 == 0:
            if np.isnan(loss):  # If loss is nan, break the loop
                break
            
    return weights, loss_history

# Set the number of iterations and learning rate
num_iterations = 10000
learning_rate = 0.001  # Smaller learning rate

# Train the model
weights, loss_history = gradient_descent(X_train, y_train, num_iterations, learning_rate)

# Prediction function
def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return probabilities >= 0.5

# Predict on test data
y_pred = predict(X_test, weights)

# Compute the confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate precision, recall, F1-score, and accuracy
precision = conf_matrix[1, 1] / sum(conf_matrix[:, 1])
recall = conf_matrix[1, 1] / sum(conf_matrix[1, :])
f1_score = 2 * precision * recall / (precision + recall)
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")

# Plot the loss over iterations
plt.plot(loss_history)
plt.title('Loss over iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
