from nose.tools import assert_equal, assert_almost_equal, assert_less_equal
from numpy.testing import assert_array_almost_equal
import numpy as np

import matplotlib.pyplot as plt

import regression

def test_generate_sin_data():
    N = 100
    x, y = regression.generate_sin_data(N)

    assert_equal(x.shape, (N,))
    assert_equal(y.shape, (N,))

    # Uncomment to plot
    plt.figure()
    plt.plot(x, y, 'b+')
    plt.savefig('/tmp/sin_data.png')


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


def test_kernel_regression_linear_data():
    # Generate data for an arbitrary line
    w = 0.5
    b = 1

    x = np.linspace(0, 1, 100)
    y = x * w + b

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = regression.partition_data(
        x[:,np.newaxis], y, val_ratio=0)

    kernel_regression = regression.KernelRegression(regression.rbf_kernel, alpha=0.1)
    kernel_regression.fit(X_tr, y_tr)

    print("Coefs: {}".format(kernel_regression.coef_))

    # assert_equal(len(kernel_regression.coef_), X_tr.shape[0])

    y_pred = kernel_regression.predict(X_test)

    plt.figure()
    plt.plot(X_tr, y_tr, 'k+')
    plt.plot(X_test, y_test, 'b+')
    plt.plot(X_test, y_pred, 'r+')
    plt.savefig('/tmp/kernel_regression_linear_data.png')

    print("Kernel regression MSE: {}".format(regression.mse(y_test, y_pred)))
    assert_less_equal(regression.mse(y_test, y_pred), 0.05)


def test_kernel_regression_sin_data():
    N = 100
    x, y = regression.generate_sin_data(
        N, theta_start=-0.1, theta_end=1.9, noise_sigma=0.3)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = regression.partition_data(
        x[:,np.newaxis], y, val_ratio=0.4)


    kernel_regression = regression.KernelRegression(regression.rbf_kernel, alpha=0.1)
    kernel_regression.fit(X_tr, y_tr)

    print("Coefs: {}".format(kernel_regression.coef_))

    # assert_equal(len(kernel_regression.coef_), X_tr.shape[0])

    y_pred = kernel_regression.predict(X_test)

    ###
    # Plot some results
    ###
    plt.figure()
    plt.plot(X_tr, y_tr, 'k+')
    plt.plot(X_test, y_test, 'b+')
    plt.plot(X_test, y_pred, 'r+')
    x_plot = np.linspace(x.min(), x.max(), 200)
    plt.plot(x_plot, kernel_regression.predict(x_plot), 'r-')
    plt.savefig('/tmp/kernel_regression_sin_data.png')

    print("Kernel regression MSE: {}".format(regression.mse(y_test, y_pred)))
    assert_less_equal(regression.mse(y_test, y_pred), 0.5)
