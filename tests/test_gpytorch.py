import pytest
import torch
import gpytorch
from regdata import Olympic, NonStat2D
from gpytorch.kernels import RBFKernel, ScaleKernel

from nsgptorch.models import GP
from nsgptorch.kernels import rbf


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@pytest.mark.eq2d
def test_equals_2d():
    n_iters = 100

    datafunc = NonStat2D(backend="torch")
    X_train, y_train, X_test = map(lambda x: x.to(torch.float32), datafunc.get_data())
    y_test = datafunc.f(X_test[:, 0], X_test[:, 1]).to(torch.float32)

    kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[1]))
    our_kernel = [rbf for _ in range(X_train.shape[1])]
    our_inducing = [None for _ in range(X_train.shape[1])]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    our_model = GP(our_kernel, X_train.shape[1], our_inducing)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    our_optimizer = torch.optim.Adam(our_model.parameters(), lr=0.1)

    for seed in range(5):

        torch.manual_seed(seed)
        for param in model.parameters():
            torch.nn.init.normal_(param)
        torch.manual_seed(seed)
        for param in our_model.parameters():
            torch.nn.init.normal_(param)

        for _ in range(n_iters):
            optimizer.zero_grad()
            our_optimizer.zero_grad()
            out = model(X_train)
            loss = -mll(out, y_train)
            our_loss = our_model(X_train, y_train)
            loss.backward()
            our_loss.backward()
            optimizer.step()
            our_optimizer.step()

        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = likelihood(model(X_test))

        our_dist = our_model.predict(X_train, y_train, X_test)

        mse = (dist.mean.ravel() - y_test.ravel()).pow_(2).mean().sqrt_()
        our_mse = (our_dist.mean.ravel() - y_test.ravel()).pow_(2).mean().sqrt_()

        assert abs(mse - our_mse) < 0.05
        # print(seed, "our", our_mse, "their", mse, "diff", abs(mse - our_mse))
