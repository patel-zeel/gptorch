## NSGP-Torch

## Examples

### `gpytorch` model with `skgpytorch`
```python
# Import packages
import torch
from regdata import NonStat2D
from gpytorch.kernels import RBFKernel, ScaleKernel
from skgpytorch.models import ExactGPRegressor
from skgpytorch.metrics import mean_squared_error

# Hyperparameters
n_iters = 100

# Load data
datafunc = NonStat2D(backend="torch")
X_train, y_train, X_test = map(lambda x: x.to(torch.float32), datafunc.get_data())
y_test = datafunc.f(X_test[:, 0], X_test[:, 1]).to(torch.float32)

# Define a kernel
kernel = ScaleKernel(RBFKernel(ard_num_dims=X_train.shape[1]))

# Define a model 
model = ExactGPRegressor(X_train, y_train, kernel, device='cpu')

# Train the model
model.fit(n_iters=n_iters, random_state=seed)

# Predict the distribution
pred_dist = model.predict(X_train, y_train, X_test)

# Compute RMSE and/or NLPD
mse = mean_squared_error(pred_dist, y_test, squared=False)
nlpd = neg_log_posterior_density(pred_dist, y_test)
```

### `nsgptorch` model with `skgpytorch`
```python
# Import packages
import torch
from regdata import NonStat2D

from nsgptorch.kernels import rbf

from skgpytorch.models import ExactNSGPRegressor
from skgpytorch.metrics import mean_squared_error

# Hyperparameters
n_iters = 100

# Load data
datafunc = NonStat2D(backend="torch")
X_train, y_train, X_test = map(lambda x: x.to(torch.float32), datafunc.get_data())
y_test = datafunc.f(X_test[:, 0], X_test[:, 1]).to(torch.float32)

# Define a kernel list for each dimension
kernel_list = [rbf, rbf]

# Define inducing points for each dimension (must be none if not applicable)
inducing_points = [None, None]

# Define a model 
model = ExactNSGPRegressor(kernel_list, input_dim=2, inducing_points, device='cpu')

# Train the model
model.fit(X_train, y_train, n_iters=n_iters, random_state=seed)

# Predict the distribution
pred_dist = model.predict(X_train, y_train, X_test)

# Compute RMSE and/or NLPD
mse = mean_squared_error(pred_dist, y_test, squared=False)
nlpd = neg_log_posterior_density(pred_dist, y_test)
```

## Plan

* Each kernel is 1D
* Multiply kernels to each other

## Ideas
* Compute distance once and save it
* Update skgpytorch to use 1 std instead of 0.1
* Do something about mean learning of gpytorch for comparison
