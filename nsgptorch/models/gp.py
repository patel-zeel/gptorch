import torch
from torch.distributions import MultivariateNormal
from ..noise import Noise
from ..scale import Scale


class GP(torch.nn.Module):
    def __init__(self, kernel_list, noise_lower_bound=1e-5, noise_constraint=None):
        super().__init__()
        assert isinstance(kernel_list, list), "kernel_list must be a list"
        self.kernel_list = torch.nn.ModuleList(kernel_list)
        self.noise = Noise(lower_bound=noise_lower_bound, constraint=noise_constraint)
        self.scale = Scale()

        self.mean = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))

    def compute_covar(self, X1, X2):
        covar = 1.0
        if self.training:
            self.additional_loss_terms = []
            for kernel in self.kernel_list:
                if kernel.is_stationary():
                    tmp_covar = kernel(X1, X2)
                else:
                    tmp_covar, local_loss = kernel(X1, X2)
                    self.additional_loss_terms.append(local_loss)

                covar = covar * tmp_covar
        else:
            for kernel in self.kernel_list:
                tmp_covar = kernel(X1, X2)
                covar = covar * tmp_covar

        # Apply signal variance
        covar = self.scale(covar)
        return covar

    def forward(self, X, y=None, X_test=None, return_cov=True):
        if self.training:
            assert y is None, "y must be None in training mode"
            assert X_test is None, "X_test must be None in training mode"

            # Compute the kernel matrix
            covar = self.compute_covar(X, X)

            # Apply noise variance
            covar = self.noise(covar)
            try:
                distr = MultivariateNormal(self.mean.expand(X.shape[0]), covar)
            except:
                raise ValueError()
            return distr
        else:
            K = self.compute_covar(X, X)
            K = self.noise(K)

            L = torch.linalg.cholesky(K)
            del K

            # Mean prediction
            alpha = torch.cholesky_solve(y.reshape(-1, 1) - self.mean, L)
            k_star = self.compute_covar(X_test, X)
            pred_mean = (k_star @ alpha) + self.mean
            del alpha
            if not return_cov:
                return pred_mean.ravel()

            # Covaariance prediction
            k_star_star = self.compute_covar(X_test, X_test)
            v = torch.cholesky_solve(k_star.T, L)
            del L
            pred_cov = k_star_star - k_star @ v
            pred_cov = self.noise(pred_cov)

            return pred_mean.ravel(), pred_cov

    def set_train_data(self, X, y, strict=False):
        # Just for skgpytorch compatibility, remove in future
        pass
