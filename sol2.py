import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the cost (mean squared error)
def calculate_cost(X, y, theta):
    predictions = np.dot(X, theta)
    errors = predictions - y
    cost = np.sum(errors ** 2) / (2 * len(y))
    return cost

# Gradient Descent Function
def gradient_descent(X, y, theta, learning_rate=0.1, stopping_threshold=1e-6, num_iterations=1808):
    costs = []
    weights = []

    for _ in range(num_iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradient = np.dot(X.T, errors) / len(y)
        theta -= learning_rate * gradient

        # Calculate and store cost at each iteration
        cost = calculate_cost(X, y, theta)
        costs.append(cost)

        # Store weights at each iteration for visualization
        weights.append(theta.copy())

        # Stopping criterion
        if len(costs) > 1 and abs(costs[-2] - costs[-1]) <= stopping_threshold:
            break

    return theta, costs, weights

# Load training data
train_data = np.genfromtxt('train.csv', delimiter=',')
X_train = train_data[:, :-1]  # Features
y_train = train_data[:, -1]   # Target

# Feature Scaling
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Add a column of ones for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Initialize weights
theta_initial = np.zeros(X_train.shape[1])

# Run Gradient Descent
theta_final, cost_history, theta_history = gradient_descent(X_train, y_train, theta_initial)

# Plotting the Regression Line along with Scatter Plot of Training Data
plt.scatter(X_train[:, 1], y_train, label='Actual Data')
plt.plot(X_train[:, 1], np.dot(X_train, theta_final), color='red', label='Final Regression Line')
plt.title('Regression Line with Training Data')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()
