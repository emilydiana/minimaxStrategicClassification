import numpy as np
from sklearn.svm import LinearSVC


class LinearSVM(LinearSVC):
    def predict(self, X):
        decision_values = self.decision_function(X)
        # Example: Custom logic could go here, for now, we'll just use the default threshold
        predictions = (decision_values >= 0).astype(int)
        return predictions
