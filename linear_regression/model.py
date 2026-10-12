import numpy as np

import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0.0
        self.cost_history = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        
        # Automatically reshape 1D feature arrays to 2D column matrices (m, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        m, n = X.shape
        
        # Initialize weights for all features and bias to zero
        self.W = np.random.randn(n, 1) * 0.01
        self.b = 0.0
        
        for epoch in range(self.epochs):
            # --- 1. Forward Pass (Matrix multiplication) ---
            y_pred = np.dot(X, self.W) + self.b
            
            # --- 2. Compute Cost (MSE) ---
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # --- 3. Compute Gradients (Vectorized) ---
            dW = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)
            
            # --- 4. Update Parameters ---
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.W) + self.b
