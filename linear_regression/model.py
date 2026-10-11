import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.cost_history = []

    def fit(self, X, y):
        m = X.shape[0]
        
        # Initialize weights randomly
        self.w = np.random.randn()
        self.b = np.random.randn()
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.w * X + self.b
            
            # Compute Cost (MSE)
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            # Compute Gradients
            dw = (1 / m) * np.sum(error * X)
            db = (1 / m) * np.sum(error)
            
            # Update Parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
        return self

    def predict(self, X):
        return self.w * X + self.b
