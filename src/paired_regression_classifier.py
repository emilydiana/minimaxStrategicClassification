# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from sklearn.linear_model import LinearRegression


class PairedRegressionClassifier:
    """
    A custom machine learning algorithm for fair classification.
    Trains two regression models for predicting positive and negative labels. To make predictions, return the label
    corresponding to the model with the higher score.
    The regression labels for
    """

    def __init__(self, regressor_class=LinearRegression):
        """
        :param regressor_class: The regression model class to use (recommended to use an sklearn class).
                                Must train with .fit(X, y) and predict with .predict(X)
                                Defaults to linear regression if no arguments are provided.
        """
        self.regressor = regressor_class()

    def fit(self, X, y, w):
        """
        Fits the model to the weighted classification problem provided.
        Sample weights are used to create new label sets for each of the classifiers.
        A positive label becomes the sample weight for the positive regressor, and a 0 for the negative regressor.
        A negative label becomes the sample weight for the negative regressor, and a 0 for the positive regressor.
        :param X: Feature matrix
        :param y: labels
        :param w: sample weights
        :return: The trained classifier
        """
        # Creates appropriate label matrices
        costs = y * w - (1 - y) * w  # Put 0s where there is a 0 in y, replace 1s with sample weight. Element wise multiply
        # Fits the two models
        self.regressor.fit(X, costs)

        return self

    def predict(self, X):
        """
        :param X: Feature matrix to make predictions on
        :return: array of predicted labels
        """
        yhat = self.regressor.predict(X)
        return yhat >= 0  # True (1) when yhat >= 0, otherwise False (0)

