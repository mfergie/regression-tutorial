"""
A simple implementation of linear regression.
"""
import math
import numpy as np

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


def rbf_kernel(X1, X2, gamma=0.1):
    """
    Computes radial basis functions between inputs in X1 and X2.

    K(x, y) = exp(-gamma ||x - y||^2)

    This is a slow implementation for illustrative purposes.
    """
    n_samples_rr = X1.shape[0]
    n_samples_cc = X2.shape[0]

    pairwise_distances = np.zeros((n_samples_rr, n_samples_cc))

    # Compute pairwise distances
    for rr in range(n_samples_rr):
        for cc in range(n_samples_cc):
            pairwise_distances[rr, cc] = np.sum((X1[rr] - X2[cc])**2)

    K = np.exp(-gamma * pairwise_distances)

    return K


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
        # Check input is the correct shape
        if X.ndim == 1:
            X = X[:,np.newaxis]

        # Append a column of 1's for bias
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        X = np.matrix(X) # Use a NumPy matrix for clarity

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
        # Check input is the correct shape
        if X.ndim == 1:
            X = X[:,np.newaxis]

        # Prepend a column of 1's onto input for intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        X = np.matrix(X) # Use a NumPy matrix for clarity

        # Access our computed model parameters
        w = np.matrix(self.coef_).T

        # Compute prediction
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

        # Add regularization to K
        K += np.identity(K.shape[0]) * self.alpha

        K = np.hstack((np.ones((K.shape[0],1)), K))

        y = np.matrix(y[:,np.newaxis])

        # Compute parameters
        ktk_inv = np.linalg.inv(K.T * K)
        params = ktk_inv * K.T * y

        # Store the parameters
        self.coef_ = np.asarray(params).flatten()

    def predict(self, X):
        """
        Predict model.
        """
        if X.ndim == 1:
            X = X[:,np.newaxis]

        # Transform inputs via kernel – Note use of
        # original training data through self.X_
        K = self.kernel_fn(X, self.X_)

        K = np.hstack((np.ones((K.shape[0],1)), K))

        w = np.matrix(self.coef_).T

        y = K * w
        
        return np.asarray(y).flatten()
