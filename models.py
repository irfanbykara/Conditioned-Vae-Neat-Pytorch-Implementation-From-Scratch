import torch
from torch.nn import Module

class VariationalAutoEncoder(Module):
    def __init__(self, latent_dim: int, encoder_module: Module, decoder_module: Module):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_module = encoder_module
        self.decoder_module = decoder_module

    def sample(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        :param mean: :math:`\mu(x)`: the mean obtained from x. Size = N x  latent_dim
        :param log_std: :math:`\Sigma(x)`: the diagonal terms for the standard deviation,
                        estimated as :math:`\log \sigma_i^2`. Size = N x  latent_dim.
        :return: Samples with Gaussian distribution :math:`\mathcal{N}(\mu(x), \Sigma(x)`
        """
        epsilon = torch.randn_like(log_std)
        z = mean + log_std * epsilon
        return z

    def forward(self, x: torch.Tensor,y:torch.Tensor):
        """
        Forward pass for the network.
        :return: A torch.Tensor.
        """

        mean, log_var = self.encoder_module(x,y)
        z = self.sample(mean, log_var)
        x_hat = self.decoder_module(z,y)
        return x,x_hat, mean, log_var
