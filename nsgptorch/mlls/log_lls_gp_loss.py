import torch


class LogLLSGPLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, distr, y):
        self.lml = distr.log_prob(y) / y.numel()
        additional_loss = 0
        for loss_term in self.model.additional_loss_terms:
            additional_loss += loss_term
        return self.lml + additional_loss
