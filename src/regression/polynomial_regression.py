import numpy as np


class PolynomialRegression:
    """
    Parameters
    ----------
    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization.
    """
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
        self.w = None

    def predict(self, X):
        return np.dot(X, self.w)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X :{array-like, sparse matrix}
            Subset of training data.
        y : numpy array
            Subset of target values.

        """
        self.w = np.dot(np.dot(np.linalg.inv(self.alpha*np.eye(X.shape[1]) + np.dot(X.T, X)), X.T), y)
