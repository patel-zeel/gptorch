import pytest
import torch
import gpytorch
from regdata import Olympic, NonStat2D
from gpytorch.kernels import RBFKernel, ScaleKernel
from skgpytorch.models import ExactGPRegressor

from nsgp_torch.models import GP
from nsgp_torch.kernels import rbf


@pytest.mark.eq1d
def test_equals_1d():
    # torch.manual_seed(0)

    X_train, y_train, X_test = Olympic(backend="torch").get_data()
    # X_train, y_train, X_test = NonStat2D(backend="torch").get_data()

    X_train = X_train.to(torch.float)
    X_test = X_test.to(torch.float)
    y_train = y_train.to(torch.float).ravel()

    kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[1]))
    model = ExactGPRegressor(X_train, y_train, kernel)
    model.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model.model)
    # for param in model.model.mean_module.parameters():
    # model.model.covar_module.base_kernel.raw
    # print(model.model.covar_module(X_train)[0, 2].item())
    dist = model.model(X_train)
    # print(model.likelihood(dist).covariance_matrix[0, 1].item())
    loss = -model.mll(dist, y_train)
    # print("Previous model", loss.item())
    n_iters = 0

    model.fit(n_iters=n_iters, n_restarts=0)

    dist = model.predict(X_test, dist_only=True)
    print("Prev-mean", dist.mean[:10])

    # print(model.history["train_loss"][-10:])
    our_model = GP([rbf], X_train.shape[1], [None])
    optim = torch.optim.Adam(our_model.parameters(), lr=0.1)
    # for param in our_model.parameters():
    # torch.nn.init.normal_(param, mean=0, std=0.1)
    losses = []
    for _ in range(n_iters):
        optim.zero_grad()
        loss = our_model(X_train, y_train)
        loss.backward()
        losses.append(loss.item())
        optim.step()
    # print(losses[-10:])
    print("prev-ls", model.model.covar_module.base_kernel.raw_lengthscale)
    print("our-ls", our_model.raw_lengthscale0.item())
    print("prev-var", model.model.covar_module.raw_outputscale.item())
    print("our-var", our_model.raw_global_variance.item())
    print("prev-noise", model.model.likelihood.raw_noise.item())
    print("our-noise", our_model.raw_global_noise_variance.item())
    print("prev-mean", model.model.mean_module.constant.item())
    print("our-mean", our_model.raw_mean.item())

    mean = our_model.predict(X_train, y_train, X_test).mean
    print("our-mean", mean[:10].ravel())
