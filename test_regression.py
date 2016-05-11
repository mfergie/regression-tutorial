from nose.tools import assert_equal, assert_almost_equal, assert_less_equal
from numpy.testing import assert_array_almost_equal
import numpy as np
from sklearn.metrics import pairwise

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
    # x, y = regression.generate_sin_data(N)

    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START, theta_end=THETA_END, noise_sigma=NOISE_SIGMA)


    assert_equal(x.shape, (N,))
    assert_equal(y.shape, (N,))

    # Uncomment to plot
    plt.figure()
    plt.plot(x, y, 'ko')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('/tmp/sin_data.png', bbox_inches="tight")


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
    # Generate data for an arbitrary line
    # x, y = regression.generate_sin_data(N)

    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START, theta_end=THETA_END, noise_sigma=NOISE_SIGMA)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = regression.partition_data(
        x[:,np.newaxis], y, val_ratio=0)

    linear_regression = regression.LinearRegression()
    linear_regression.fit(X_tr, y_tr)

    y_pred = linear_regression.predict(X_test)
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


def test_rbf():
    """
    Test against sklearn implementation.
    """
    X1 = np.random.random_sample((10, 5))
    X2 = np.random.random_sample((8, 5))

    K_sk = pairwise.rbf_kernel(X1, X2, gamma=0.1)
    K_ours = regression.rbf_kernel(X1, X2, gamma=0.1)

    assert_array_almost_equal(K_sk, K_ours)


def test_kernel_regression_sin_data():
    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START, theta_end=THETA_END, noise_sigma=NOISE_SIGMA)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = regression.partition_data(
        x[:,np.newaxis], y, val_ratio=0.4)


    kernel_regression = regression.LinearRegression()

    K = regression.rbf_kernel(X_tr, X_tr)
    K += np.identity(K.shape[0]) * 0.1

    kernel_regression.fit(K, y_tr)

    print("Coefs: {}".format(kernel_regression.coef_))

    # assert_equal(len(kernel_regression.coef_), X_tr.shape[0])

    K_test = regression.rbf_kernel(X_test, X_tr)
    y_pred = kernel_regression.predict(K_test)

    ###
    # Plot some results
    ###
    x_plot = np.linspace(x.min(), x.max(), 200)
    K_plot = regression.rbf_kernel(x_plot[:,np.newaxis], X_tr)
    y_plot = kernel_regression.predict(K_plot)

    regression.plot_figure(
        X_tr,
        y_tr,
        x_plot,
        y_plot,
        filename='/tmp/kernel_regression_sin_data.png')


    print("Kernel regression MSE (sin): {}".format(regression.mse(y_test, y_pred)))
    assert_less_equal(regression.mse(y_test, y_pred), 0.5)
