
import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class VAELoss(_Loss):
    def __init__(self,beta=None):
        super(VAELoss, self).__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor,mean:torch.Tensor,log_var:torch.Tensor) -> torch.Tensor:
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Total VAE loss
        if self.beta != None:
            total_loss = reconstruction_loss + self.beta * kl_divergence
        else:
            total_loss = reconstruction_loss + kl_divergence

        return total_loss

