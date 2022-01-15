import torch
from ..kernels.rbf import rbf, rbf_init
from ..kernels.transforms import log_exp

from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.distributions import MultivariateNormal


class GP(torch.nn.Module):
    def __init__(self, kernel_list, input_dim, inducing_points):
        super(GP, self).__init__()
        assert len(kernel_list) == input_dim
        assert len(inducing_points) == input_dim

        self.kernel_list = kernel_list

        self.register_buffer("zero", torch.zeros(1, dtype=torch.float))
        self.register_buffer("one", torch.ones(1, dtype=torch.float))

        # Initialize parameters
        self.raw_global_variance = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float)
        )
        self.raw_global_noise_variance = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float)
        )
        self.raw_mean = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        for k_i, kernel in enumerate(kernel_list):
            if kernel.__name__ == "rbf":
                rbf_init(self, k_i)
                assert inducing_points[k_i] is None
            else:
                pass
                # TODO: Register inducing points as parameters

    def compute_kern(self, X1, X2):
        kern = (log_exp(self.raw_global_variance) * self.one).repeat(
            (X1.shape[0], X2.shape[0])
        )

        for k_i, kernel in enumerate(self.kernel_list):
            if kernel.__name__ == "rbf":
                kern.mul_(
                    kernel(
                        X1[:, k_i],
                        X2[:, k_i],
                        self.get_parameter(f"raw_lengthscale{k_i}"),
                    )
                )
        return kern

    def forward(self, X, y):
        kern = self.compute_kern(X, X)

        kdiag = kern.diagonal()
        kdiag += log_exp(self.raw_global_noise_variance)

        # dist = MultivariateNormal(self.zero.repeat(X.shape[0]), kern)
        dist = MultivariateNormal(self.raw_mean.repeat(X.shape[0]), kern)
        return -dist.log_prob(y) / X.numel()

    def predict(self, X_orig, y_orig, X_pred):
        with torch.no_grad():
            kern = self.compute_kern(X_orig, X_orig)
            kdiag = kern.diagonal()
            kdiag += log_exp(self.raw_global_noise_variance)
            l_kern = psd_safe_cholesky(kern)
            alpha = torch.cholesky_solve(y_orig.reshape(-1, 1) - self.raw_mean, l_kern)
            del kern
            k_star = self.compute_kern(X_pred, X_orig)
            k_star_star = self.compute_kern(X_pred, X_pred)
            mean = k_star @ alpha + self.raw_mean
            v = torch.cholesky_solve(k_star.T, l_kern)
            cov = k_star_star - k_star @ v
            kdiag = cov.diagonal()
            kdiag += log_exp(self.raw_global_noise_variance)

            pred_dist = MultivariateNormal(mean.ravel(), cov)
            return pred_dist
