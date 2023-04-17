from sklearn.utils import shuffle
import numpy as np


class MBGDRegressor:

    """
    Parameters
    ----------
    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization.

    eta0 : float, default=0.001
        The initial learning rate .

    power_t : float, default=0.25
        The exponent for inverse scaling learning rate.
    """

    def __init__(self, alpha=0.0001, eta0=0.001, power_t=0.5):
        self.alpha = alpha
        self.eta0 = eta0
        self.power_t = power_t
        self.w = None
        self.loss_list = []

    def _loss(self, y, X):
        loss = 0
        loss += np.sum(np.square(self.predict(X) - y))
        loss += self.alpha * np.sum(np.square(self.w))
        return loss / X.shape[1]

    def _gradient(self, y, y_pred, X):
        g = 2 * (np.dot((y_pred - y), X) + np.dot(self.alpha, self.w)) / y.shape[0]
        return g

    def predict(self, X):
        return np.dot(X, self.w)

    def fit(self, X, y, init_w=None, random_state=42, batch_size=32, epoch=100):
        """
        Parameters
        ----------
        X : array-like
            Subset of training data;

        y : numpy array
            Subset of target values;

        init_w : array-like
            Initial weights;

        random_state : int, default=42
            Random state;

        batch_size : int, default=32
            Size of one batch;

        epoch : int, default=100
            Maximum number of passes over the training data.
        """
        if init_w is None and self.w is None:
            self.w = np.ones(X.shape[1], 1)
        else:
            self.w = init_w

        np.random.seed(random_state)
        for i in range(epoch):
            X, y = shuffle(X, y, random_state=random_state)
            cur_pos = 0
            while cur_pos < X.shape[0]:
                X_part = X[cur_pos:min(cur_pos + batch_size, X.shape[0] - 1)]
                y_part = y[cur_pos:min(cur_pos + batch_size, X.shape[0] - 1)]
                step = self.eta0 / pow(i * X.shape[1] / 32 + 1, self.power_t)
                self.w = self.w - step * self._gradient(y_part, self.predict(X_part), X_part)
                cur_pos += batch_size

            cur_loss = self._loss(y, X)
            self.loss_list.append(cur_loss)
            print(f'current mse: {cur_loss}')
