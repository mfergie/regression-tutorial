from nose.tools import assert_equal, assert_almost_equal, assert_less_equal
from numpy.testing import assert_array_almost_equal
import numpy as np

import matplotlib.pyplot as plt

import regression
import math

N=30
THETA_START=-1.9
THETA_END=2.2
NOISE_SIGMA=0.4

# N=50
# THETA_START=-2*math.pi
# THETA_END=2*math.pi
# NOISE_SIGMA=0.4


def test_generate_sin_data():
    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START, theta_end=THETA_END, noise_sigma=NOISE_SIGMA)


    assert_equal(x.shape, (N,))
    assert_equal(y.shape, (N,))


def test_linear_regression():
    # Generate data for an arbitrary line
    w = 0.5
    b = 1

    x = np.linspace(0, 1, 100)
    y = x * w + b

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = regression.partition_data(
        x[:,np.newaxis], y, val_ratio=0)

    linear_regression = regression.LinearRegression()
    linear_regression.fit(X_tr, y_tr)

    assert_almost_equal(linear_regression.coef_[0], b)
    assert_almost_equal(linear_regression.coef_[1], w)

    y_pred = linear_regression.predict(X_test)

    assert_array_almost_equal(y_pred, y_test)
    assert_almost_equal(regression.mse(y_test, y_pred), 0)


def test_linear_regression_sin_data():

    # Generate some data to test with
    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N,
        theta_start=THETA_START,
        theta_end=THETA_END,
        noise_sigma=NOISE_SIGMA)

    # Partition the training data into train/test sets
    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = (
        regression.partition_data(x[:,np.newaxis], y, val_ratio=0))

    # Create LinearRegression model
    linear_regression = regression.LinearRegression()

    # Fit it to our training data
    linear_regression.fit(X_tr, y_tr)

    # Predict for test inputs
    y_pred = linear_regression.predict(X_test)

    # Test that we achieve reasonable prediction accuracy
    assert_less_equal(regression.mse(y_test, y_pred), 1.2)
    print("Linear MSE: {}".format(regression.mse(y_test, y_pred)))

    x_plot = np.linspace(x.min(), x.max(), 200)
    y_plot = linear_regression.predict(x_plot)

    regression.plot_figure(
        X_tr,
        y_tr,
        x_plot,
        y_plot,
        filename='/tmp/linear_regression_sin_data.png')


###
# Test below commented out to avoid sklearn dependency.
###

# def test_rbf():
#     """
#     Test against sklearn implementation.
#     """
#     from sklearn.metrics import pairwise
#     X1 = np.random.random_sample((10, 5))
#     X2 = np.random.random_sample((8, 5))
#
#     K_sk = pairwise.rbf_kernel(X1, X2, gamma=0.1)
#     K_ours = regression.rbf_kernel(X1, X2, gamma=0.1)
#
#     assert_array_almost_equal(K_sk, K_ours)


def test_kernel_regression_sin_data():
    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START,
        theta_end=THETA_END, noise_sigma=NOISE_SIGMA)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = (
        regression.partition_data(x[:,np.newaxis], y, val_ratio=0))

    # Create model
    kernel_regression = regression.LinearRegression(alpha=1e-5)

    # Define our kernel function.
    def kernel_fn(X):
        # Bind the RBF kernel to use the training inputs
        return regression.rbf_kernel(X, X_tr)

    # Compute the kernel
    K = kernel_fn(X_tr)

    kernel_regression.fit(K, y_tr)

    K_test = kernel_fn(X_test)
    y_pred = kernel_regression.predict(K_test)

    print("Kernel regression MSE: {}".format(regression.mse(y_test, y_pred)))
    assert_less_equal(regression.mse(y_test, y_pred), 0.8)

    # Plot output
    x_plot = np.linspace(x.min(), x.max(), 200)
    K_plot = kernel_fn(x_plot[:,np.newaxis])
    y_plot = kernel_regression.predict(K_plot)

    regression.plot_figure(
        X_tr,
        y_tr,
        x_plot,
        y_plot,
        filename='/tmp/kernel_regression_sin_data.png')
