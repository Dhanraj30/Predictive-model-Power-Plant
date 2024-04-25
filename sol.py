import numpy as np

import matplotlib.pyplot as plt
# Load training data
train_data = np.genfromtxt('train.csv', delimiter=',')
X_train = train_data[:, :-1]  # Features
y_train = train_data[:, -1]   # Target

# Feature Scaling
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Add a column of ones for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Initialize weights
theta = np.zeros(X_train.shape[1])

# Hyperparameters
learning_rate = 0.8178
num_iterations = 1000

# Gradient Descent
for _ in range(num_iterations):
    predictions = np.dot(X_train, theta)
    errors = predictions - y_train
    gradient = np.dot(X_train.T, errors) / len(y_train)
    theta -= learning_rate * gradient

# Load test data
X_test = np.genfromtxt('test.csv', delimiter=',')
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Make predictions
predictions = np.dot(X_test, theta)

# Save predictions to a CSV file
np.savetxt('predictions.csv', predictions, delimiter=',', fmt='%.2f', header='predictions', comments='')

# Calculate the mean of the actual values
mean_y = np.mean(y_train)

# Calculate the total sum of squares
total_sum_squares = np.sum((y_train - mean_y) ** 2)

# Calculate the residual sum of squares
residual_sum_squares = np.sum((y_train - np.dot(X_train, theta)) ** 2)

# Calculate R^2
r_squared = 1 - (residual_sum_squares / total_sum_squares)

print("Coefficient of Determination (R^2):", r_squared)




