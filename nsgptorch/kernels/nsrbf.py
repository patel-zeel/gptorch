import torch
from ..distance import sq_euclidean_distance
from ..transforms import LogExp
from ..models import GP


class NSRBF(torch.nn.Module):
    def __init__(
        self,
        X_inducing,
        base_kernel,
        constraint=None,
        active_dim: int = None,
        local_noise_bool=True,
        noise_lower_bound=1e-5,
        noise_constraint=None,
    ):
        super().__init__()
        self.register_buffer("X_inducing", X_inducing.reshape(-1, 1))
        self.raw_lengthscale_inducing = torch.nn.Parameter(
            torch.ones_like(self.X_inducing)
        )
        if constraint is None:
            self.constraint = LogExp()
        else:
            self.constraint = constraint

        self.active_dim = active_dim
        # Register local GP
        base_kernel.active_dim = 0  # Because this is always a 1d kernel
        self.local_gp = GP(
            [base_kernel],
            noise_lower_bound=noise_lower_bound,
            noise_constraint=noise_constraint,
        )
        if not local_noise_bool:
            self.local_gp.noise.raw_noise.requires_grad_(False)

    @property
    def lengthscale_inducing(self):
        self.constraint.transform(self.raw_lengthscale_inducing)

    @lengthscale_inducing.setter
    def lengthscale_inducing(self, x):
        with torch.no_grad():
            val = self.constraint.inverse_transform(torch.as_tensor(x))
            self.raw_lengthscale_inducing.fill_(val)

    def forward(self, x1, x2):
        assert self.active_dim is not None, "active_dim must be specified"
        x1 = x1[:, self.active_dim].reshape(-1, 1)
        x2 = x2[:, self.active_dim].reshape(-1, 1)
        if self.training:
            assert torch.equal(x1, x2), "x1 and x2 must be the same in training mode"
            l_mean, l_cov = self.get_local_pred(x1, mean_only=False, transform=False)
            local_pred_distr = torch.distributions.MultivariateNormal(l_mean, l_cov)
            local_log_prob = local_pred_distr.log_prob(l_mean) / l_mean.numel()
            del local_pred_distr

            l_mean = self.constraint.transform(l_mean)
            covar = (l_mean.reshape(-1, 1) * l_mean.reshape(1, -1)) ** 0.5
            l1l2_sq = (l_mean.reshape(-1, 1) ** 2) + (l_mean.reshape(1, -1) ** 2)
            covar.div_(l1l2_sq**0.5).mul_(
                (-sq_euclidean_distance(x1, x2).div_(l1l2_sq)).exp_()
            )
            del l1l2_sq
            return covar * (2**0.5), local_log_prob
        else:
            l1_mean = self.get_local_pred(x1, mean_only=True, transform=True)
            l2_mean = self.get_local_pred(x2, mean_only=True, transform=True)

            covar = (l1_mean.reshape(-1, 1) * l2_mean.reshape(1, -1)) ** 0.5
            l1l2_sq = (l1_mean.reshape(-1, 1) ** 2) + (l2_mean.reshape(1, -1) ** 2)
            covar.div_(l1l2_sq**0.5).mul_(
                (-sq_euclidean_distance(x1, x2).div_(l1l2_sq)).exp_()
            )
            del l1l2_sq
            return covar * (2**0.5)

    def is_stationary(self):
        return False

    def get_local_pred(self, x, mean_only=True, transform=True):
        if transform:
            transform_fn = self.constraint.transform
        else:
            transform_fn = lambda x: x

        self.local_gp.eval()
        if mean_only:
            raw_pred_mean = self.local_gp(
                self.X_inducing, self.raw_lengthscale_inducing, x, return_cov=False
            )
            return transform_fn(raw_pred_mean)
        else:
            raw_pred_mean, raw_pred_cov = self.local_gp(
                self.X_inducing, self.raw_lengthscale_inducing, x, return_cov=True
            )
        return transform_fn(raw_pred_mean), transform_fn(raw_pred_cov)
