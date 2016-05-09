"""
A simple implementation of linear regression.
"""
import math
import numpy as np

from sklearn.metrics import pairwise

import matplotlib.pyplot as plt


def generate_sin_data(n_points,
                      theta_start=0,
                      theta_end=2*math.pi,
                      noise_sigma=0.1):
    """
    Generates some test data from a sin wave with additive Gaussian noise.

    Parameters
    ----------
    n_points: int
        The number of points to generate
    start_theta: float
        Where to start on the sin function.
    end_theta: float
        Where to start on the sin function.
    noise_sigma: float
        Standard deviation of noise to add

    Returns
    -------
    X: ndarray, (N,)
        The input points.
    y: ndarray, (N,)
        The output points.
    """
    x = np.linspace(theta_start, theta_end, n_points)
    y = np.sin(x) + (np.random.randn(n_points) * noise_sigma**2)

    return x, y


def partition_data(X, y, train_ratio=0.6, val_ratio=0.5):
    """
    Partitions data into training, test and validation sets.

    Parameters
    ----------
    X: ndarray, (N, D)
        X points.
    y: ndarray (N,)
        y points.
    train_ratio: float
        Amount of data to use for training
    val_ratio: float
        The ratio of data to use for validation set after the training data has
        been removed.

    Returns
    -------

    training_set, validation_set, test_set
    """
    n_points = y.size
    randind = list(range(n_points))
    np.random.shuffle(randind)

    train_ind = round(train_ratio * n_points)
    val_ind = round(val_ratio * (n_points - train_ind)) + train_ind

    train_inds = randind[:train_ind]
    val_inds = randind[train_ind:val_ind]
    test_inds = randind[val_ind:]

    partitioned_data = (
        (X[train_inds], y[train_inds]),
        (X[val_inds], y[val_inds]),
        (X[test_inds], y[test_inds]))

    return partitioned_data

def mse(y, y_pred):
    """
    Computes the mean squared error between two sets of points.
    """
    return np.sum((y - y_pred)**2)


def rbf_kernel(X1, X2):
    """
    Computes radial basis functions between inputs in X1 and X2.
    """
    return pairwise.rbf_kernel(X1, X2)


class LinearRegression():
    """
    Implements basic linear regression.
    """

    def __init__(self):
        pass


    def fit(self, X, y):
        """
        Fit model.
        """
        if X.ndim == 1:
            X = X[:,np.newaxis]
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # Append a column of 1's for bias
        X = np.matrix(X)

        assert y.ndim == 1, "Only supports 1D y"
        y = np.matrix(y[:,np.newaxis])

        # Compute parameters
        xtx_inv = np.linalg.inv(X.T * X)
        coef = xtx_inv * X.T * y

        self.coef_ = np.asarray(coef).flatten()

    def predict(self, X):
        """
        Predict model.
        """
        if X.ndim == 1:
            X = X[:,np.newaxis]
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X = np.matrix(X)

        w = np.matrix(self.coef_).T

        y = X * w

        return np.asarray(y).flatten()


class KernelRegression():
    """
    Implements linear regression with a kernel function to allow for non-linear
    mappings.
    """

    def __init__(self, kernel_fn, alpha=1):
        self.kernel_fn = kernel_fn
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit model.
        """
        assert y.ndim == 1, "Only supports 1D y"

        ###
        # Prepare X
        ###
        if X.ndim == 1:
            X = X[:,np.newaxis]
        self.X_ = X # Save the training data this time

        # Transform inputs via kernel
        K = self.kernel_fn(X, X)
        K = np.matrix(K)

        ###
        # Prepare y
        ###
        self.y_bar_ = y.mean()
        y = y - self.y_bar_
        y = np.matrix(y[:,np.newaxis])

        # Add regularization to K
        K += np.identity(K.shape[0]) * self.alpha

        # Compute parameters
        xtx_inv = np.linalg.inv(K.T * K)
        params = xtx_inv * K.T * y

        self.coef_ = np.asarray(params).flatten()

    def predict(self, X):
        """
        Predict model.
        """
        if X.ndim == 1:
            X = X[:,np.newaxis]

        # Transform inputs via kernel – Note use of original training data
        # through self.X_
        Xk = self.kernel_fn(X, self.X_)

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X = np.matrix(X)


        w = np.matrix(self.coef_).T

        y = Xk * w

        y += self.y_bar_

        return np.asarray(y).flatten()
