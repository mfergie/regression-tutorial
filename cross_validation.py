"A script to demonstrate cross validation"

import numpy as np
import matplotlib.pyplot as plt
import regression

N=100
THETA_START=-1.9
THETA_END=2.2
NOISE_SIGMA=0.4

def fit_model(train_data, eval_data, model):
    X_tr, y_tr = train_data
    X_eval, y_eval = eval_data

    # Define our kernel function.
    def kernel_fn(X):
        # Bind the RBF kernel to use the training inputs
        return regression.rbf_kernel(X, X_tr)

    # Compute the kernel
    K = kernel_fn(X_tr)

    # Fit model
    model.fit(K, y_tr)

    # Predict the evaluation data
    K_eval = kernel_fn(X_eval)
    y_pred = model.predict(K_eval)

    return regression.mse(y_eval, y_pred)

if __name__ == "__main__":
    np.random.seed(1)
    x, y = regression.generate_sin_data(
        N, theta_start=THETA_START,
        theta_end=THETA_END, noise_sigma=NOISE_SIGMA)

    train_data, val_data, test_data = (
        regression.partition_data(x[:,np.newaxis], y, val_ratio=0.5))

    alpha_parameters = np.logspace(1, -5, 7)
    val_mse = []
    models = []

    for alpha in alpha_parameters:
        # Create model
        model = regression.LinearRegression(alpha=alpha)
        models.append(model)

        # Fit model and record accuracy
        mse = fit_model(train_data, val_data, model)
        val_mse.append(mse)

    ###
    # Now pick our best model and evaluate on test data
    ###
    argmin = np.argmin(val_mse)
    min_alpha = alpha_parameters[argmin]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.invert_xaxis()
    plt.plot(alpha_parameters, val_mse, 'b-')
    plt.axvline(min_alpha, color="r")
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.savefig('/tmp/cross-validation.png', bbox_inches="tight")

    model = regression.LinearRegression(alpha=min_alpha)

    # Now evalute with the test data
    test_mse = fit_model(train_data, test_data, model)
    print("Using alpha: {}. Test MSE: {}".format(min_alpha, test_mse))

    # And plot results
    X_tr, y_tr = train_data
    x_plot = np.linspace(x.min(), x.max(), 200)
    K_plot = regression.rbf_kernel(x_plot[:,np.newaxis], X_tr)
    y_plot = model.predict(K_plot)

    regression.plot_figure(
        X_tr,
        y_tr,
        x_plot,
        y_plot,
        filename='/tmp/cross_validation_results.png')

    plt.show()
