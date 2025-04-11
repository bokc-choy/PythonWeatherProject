import unittest
import numpy as np
from algorithms import CustomTempPredictor, custom_clustering, detect_anomalies

class TestAlgorithms(unittest.TestCase):

    def test_custom_temp_predictor(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = CustomTempPredictor(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertTrue(np.allclose(predictions, y, atol=1))

    def test_custom_clustering(self):
        data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
        labels, centers = custom_clustering(data, n_clusters=2)
        self.assertEqual(len(labels), len(data))
        self.assertTrue(set(labels).issubset({0, 1}))
        self.assertEqual(centers.shape, (2, 2))  # 2 clusters, 2 features

    def test_detect_anomalies(self):
        series = np.array([10, 12, 11, 13, 15, 100, 14, 13, 12, 15])
        anomalies = detect_anomalies(series, window_size=3, threshold=2.0)
        self.assertEqual(len(anomalies), len(series))
        self.assertTrue(anomalies[5])  # Expecting 100 to be an anomaly

if __name__ == '__main__':
    unittest.main()
