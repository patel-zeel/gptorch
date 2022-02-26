from time import time
import os
import pytest
import torch

import numpy as np
from skgpytorch.models import BaseRegressor, ExactGPRegressor
from regdata import Olympic, NonStat2D, SineJump1D
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.metrics import mean_squared_error
import matplotlib.pyplot as plt
from nsgptorch.models import GP
from nsgptorch.mlls import LogLLSGPLoss
from nsgptorch.kernels import RBF, NSRBF
from nsgptorch.inducing_helper import get_cluster_centers

torch.autograd.set_detect_anomaly(True)


class ExactNSGPRegressor(BaseRegressor):
    """
    Exact NSGP Regressor
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, kernel: list):
        self.train_x = train_x
        self.train_y = train_y
        model = GP(kernel)
        mll = LogLLSGPLoss(model)
        super().__init__(train_x, train_y, mll)

    def predict(self, test_x):
        assert self.training is False, "prediction must be done in eval mode"
        return self.mll.model(self.train_x, self.train_y, test_x)


@pytest.mark.eq2d
def test_equals_2d():
    n_iters = 6
    seed = 0
    device = "cuda"

    datafunc = NonStat2D(backend="torch")
    # datafunc = Olympic(backend="torch")
    # datafunc = SineJump1D(backend="torch")
    X_train, y_train, X_test = map(
        lambda x: x.to(torch.float32).to(device), datafunc.get_data()
    )
    y_test = datafunc.f(X_test[:, 0].cpu(), X_test[:, 1].cpu()).float().to(device)
    # torch.manual_seed(seed)
    # X_train = torch.rand(10, 2).float().to(device)
    # y_train = torch.rand(10).float().to(device)
    # X_test = torch.rand(25, 2).float().to(device)
    # y_test = torch.rand(25).float().to(device)

    nschk0 = time()
    n_inducing_points = 5
    x_inducing_list = []
    for i in range(X_train.shape[1]):
        tmp_x = np.unique(X_train[:, i].cpu().numpy())
        tmp_c = get_cluster_centers(tmp_x, n_inducing_points, random_state=seed)
        x_inducing_list.append(torch.tensor(tmp_c, dtype=X_train.dtype).to(device))
    our_ns_kernel = [
        NSRBF(ip, RBF(), active_dim=i)
        for i, ip in zip(range(X_train.shape[1]), x_inducing_list)
    ]
    our_ns_model = ExactNSGPRegressor(X_train, y_train, our_ns_kernel).to(device)
    our_ns_model.train()
    our_ns_model.fit(n_epochs=n_iters, random_state=seed)

    nschk1 = time()

    our_ns_model.eval()
    with torch.no_grad():
        pm, pv = our_ns_model.predict(X_test)
    our_ns_dist = torch.distributions.MultivariateNormal(pm, pv)
    nschk2 = time()

    ochk0 = time()
    our_kernel = [RBF(active_dim=i) for i in range(X_train.shape[1])]
    our_model = ExactNSGPRegressor(X_train, y_train, our_kernel).to(device)
    our_model.train()
    our_model.fit(n_epochs=n_iters, random_state=seed)

    ochk1 = time()

    our_model.eval()
    with torch.no_grad():
        pm, pv = our_model.predict(X_test)
    our_dist = torch.distributions.MultivariateNormal(pm, pv)
    ochk2 = time()
    # mse = (dist.mean.ravel() - y_test.ravel()).pow_(2).mean().sqrt_()
    # our_mse = (our_dist.mean.ravel() - y_test.ravel()).pow_(2).mean().sqrt_()

    gchk0 = time()
    kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[1]))
    model = ExactGPRegressor(X_train, y_train, kernel).to(device)
    model.train()
    model.fit(n_epochs=n_iters, random_state=seed)

    gchk1 = time()

    model.eval()
    with torch.no_grad():
        dist = model.predict(X_test)

    gchk2 = time()

    plt.plot(model.history["epoch_loss"][0], label="GPyTorch")
    plt.plot(our_model.history["epoch_loss"][0], "--", label="NSGP rbf")
    plt.plot(our_ns_model.history["epoch_loss"][0], "--", label="NSGP nsrbf")
    plt.legend()
    plt.title(
        f"g_ttime: {gchk1 - gchk0:.2f} s, g_ptime: {gchk2 - gchk1:.2f} s\
        \n o_ttime: {ochk1 - ochk0:.2f} s, o_ptime: {ochk2 - ochk1:.2f} s\
        \n ns_ttime: {nschk1 - nschk0:.2f} s, ns_ptime: {nschk2 - nschk1:.2f} s\
        \n rmse: {mean_squared_error(dist, y_test, squared=False):.2f},\
        our_rmse: {mean_squared_error(our_dist, y_test, squared=False):.2f},\
        our_ns_rmse: {mean_squared_error(our_ns_dist, y_test, squared=False):.2f}"
    )
    plt.tight_layout()

    if not os.path.exists("tests/fig/"):
        os.makedirs("tests/fig/")
    plt.savefig(f"tests/fig/test_equals_2d_{seed}.png")

    for pdn, pd in zip(["gpytorch", "rbf", "nsrbf"], [dist, our_dist, our_ns_dist]):
        plt.figure()
        d = int(X_test.shape[0] ** 0.5)
        x1, x2 = X_test[:, 0].cpu().numpy(), X_test[:, 1].cpu().numpy()
        plt.contourf(
            x1.reshape(d, d),
            x2.reshape(d, d),
            pd.mean.cpu().numpy().reshape(d, d),
            label=pdn,
        )
        plt.savefig(f"tests/fig/test_equals_2d_{pdn}_{seed}.png")
    # assert abs(mse - our_mse) < 0.05
    # print(seed, "our", our_mse, "their", mse, "diff", abs(mse - our_mse))
