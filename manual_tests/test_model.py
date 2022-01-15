from nsgp_torch.models import GP
from nsgp_torch.kernels import rbf

model = GP([rbf], 1, [None])
print("Done")
