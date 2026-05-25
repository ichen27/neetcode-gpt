import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        preds = []

        for i in range(len(X)):
            temp = 0
            for j in range(len(X[i])):
                temp += X[i][j] * weights[j]

            preds.append(round(temp, 5))
        
        return preds


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = 0
        for i in range(len(model_prediction)):
            for j in range(len(model_prediction[i])):
                error += np.power((model_prediction[i][j] - ground_truth[i][j]), 2)

        error /= len(model_prediction)

        return round(error, 5)





