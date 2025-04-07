import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class CustomTempPredictor(BaseEstimator, RegressorMixin):
    """
    A custom temperature predictor using gradient descent linear regression.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTempPredictor':
        """
        Fit the linear model to the data using gradient descent.
        
        Parameters:
        - X: np.ndarray of shape (n_samples, n_features)
        - y: np.ndarray of shape (n_samples)
        
        Returns:
        - self: the fitted estimator
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute gradients for weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predit using the linear model.

        Parameters:
        - X: np.ndarray of shape (n_samples, n_features)

        Returns:
        - Predictions: np.ndarray of shape (n_samples)
        """

        return np.dot(X, self.weights) + self.bias
    
def custom_cluserting(data: np.ndarray, n_clusters: int, n_iterations: int = 100) -> np.ndarray:
    """
    A simple custom clusterng algorithm similar to k-means.

    Parameters:
    - data: np.ndarray of shape (n_samples, n_features)
    - n_clusters: int, the number of clusters
    - n_iterations: int, maximum number of iterations

    Returns:
    - labels: np.ndarray of shape (n_samples), cluster labels for each data point.
    """
    n_samples, n_features = data.shape
    # Initialize cluster centers by randomly selecting data points
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centers = data[indices]
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(n_iterations):
        # Assignment step: assign each point to the nearest center
        new_labels = np.array([
            np.argmin([np.linalg.norm(point - center) for center in centers])
            for point in data
        ])
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        # Update step: recalculate centers as the mean of points in each cluster
        for k in range(n_clusters):
            if np.any(labels == k):
                centers[k] = data[labels == k].mean(axis=0)
            else:
                # If no points are assigned to the cluser, reinitialize the center randomly
                centers[k] = data[rng.choice(n_samples)]
    return labels

def detect_anomalies(time_series: np.ndarray, window_size: int = 10, threshold: float = 2.0) -> np.ndarray:
    """
    Detect anomalies in the time series data using a moving window approach.
    A point is flagged as an anomaly if it deviates from the window's moving average by more than 
    a specific threshold times the standard deviation.

    Parameters:
    - time_series: np.ndarray of shape (n_samples)
    - window_size: int, the size of the moving window
    - threshold: float, the multiplier for the standard deviation to decide anomaly

    Returns:
    - anomalies: np.ndarray of shape (n_samples), boolean array where true indicates an anomaly.
    """
    anomalies = np.zeros(len(time_series), dtype=bool)

    # Start by checking from the index equal to window_size
    for i in range(window_size, len(time_series)):
        window = time_series[i-window_size:i]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            continue # Avoid division by zero if the window is constant
        if abs(time_series[i] - mean) > threshold * std:
            anomalies[i] = True
    return anomalies

# Example usage for testing purposes
if __name__ == "__main__":
    # CustomTempPredictor example
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    predictor = CustomTempPredictor(learning_rate=0.01, n_iterations=1000)
    predictor.fit(X, y)
    predictions = predictor.predict(X)
    print("predictions:", predictions)

    # custom_clustering example
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [10, 2], [10, 4], [10, 0]])
    labels = custom_cluserting(data, n_clusters=2)
    print("Cluster labels:", labels)

    # detect_anomalies example
    time_series = np.array([10, 12, 11, 13, 15, 100, 14, 13, 12, 15, 16])
    anomalies = detect_anomalies(time_series, window_size=3, threshold=2.0)
    print("Anomalies detected:", anomalies)