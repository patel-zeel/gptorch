import pytest
import torch
import gpytorch
from regdata import Olympic, NonStat2D
from gpytorch.kernels import RBFKernel, ScaleKernel
from skgpytorch.models import ExactGPRegressor, ExactNSGPRegressor
from skgpytorch.metrics import mean_squared_error

from nsgptorch.models import GP
from nsgptorch.kernels import rbf


@pytest.mark.eq2d
def test_equals_2d():
    n_iters = 100

    datafunc = NonStat2D(backend="torch")
    X_train, y_train, X_test = map(lambda x: x.to(torch.float32), datafunc.get_data())
    y_test = datafunc.f(X_test[:, 0], X_test[:, 1]).to(torch.float32)

    kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[1]))
    our_kernel = [rbf for _ in range(X_train.shape[1])]
    our_inducing = [None for _ in range(X_train.shape[1])]

    model = ExactGPRegressor(X_train, y_train, kernel)
    our_model = ExactNSGPRegressor(our_kernel, X_train.shape[1], our_inducing)

    for seed in range(10, 15):
        model.fit(n_iters=n_iters, random_state=seed)
        our_model.fit(X_train, y_train, n_iters=n_iters, random_state=seed)

        dist = model.predict(X_train, y_train, X_test)
        our_dist = our_model.predict(X_train, y_train, X_test)

        mse = mean_squared_error(dist, y_test, squared=False)
        our_mse = mean_squared_error(our_dist, y_test, squared=False)

        assert abs(mse - our_mse) < 0.01
        # print(seed, "our", our_mse, "their", mse, "diff", abs(mse - our_mse))
