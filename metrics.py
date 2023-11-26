from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from typing import List
from torchvision.models import inception_v3
from torchvision.utils import save_image


class Metric(ABC):
    """Abstract metric class to evaluate generative models.
    """
    def __init__(self, name: str,num_samples:int=10,latent_dim:int=64, use_cuda: bool = False,) -> None:
        self.name = name
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.inception = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)

    def compute_metric(self, data_loader: DataLoader, vae: torch.nn.Module,
                       conditions: torch.Tensor = None) -> torch.Tensor:
        cumulative_score = 0.
        n_batch = 0.
        with torch.no_grad():
            for i, (x, attr) in enumerate(data_loader):
                # Move data to the device (cuda or cpu)
                x = x.to(self.device)
                attr = attr.to(self.device)

                out = self.batch_compute([x, attr], vae,conditions)

                cumulative_score += out
                n_batch += 1.

            res = cumulative_score / n_batch

        return res

    @abstractmethod
    def batch_compute(self, inp: List[torch.Tensor], vae: torch.nn.Module,conditions:torch.Tensor=None):
        raise NotImplementedError


class InceptionScore(Metric):
    # TODO implementation of InceptionScore
    def __init__(self,num_samples: int = 10,latent_dim: int = 64, use_cuda: bool = False):
        super().__init__("Inception Score",num_samples=num_samples,latent_dim=latent_dim,use_cuda=use_cuda)

    def batch_compute(self, inp: torch.Tensor, vae: torch.nn.Module,conditions: torch.Tensor = None):

        assert len(inp) > 0, "Input list is empty."

        vae = vae.to(self.device)
        conditions = conditions.to(self.device)

        random_latents = torch.randn(self.num_samples, self.latent_dim,device=self.device)
        with torch.no_grad():
            generated_images = vae.decoder_module(random_latents,conditions)
        save_image(generated_images, 'generated_images.png', nrow=10)

        images = torch.cat([generated_images] * 3, dim=1)

        # Resize images to Inception model's input size (299x299)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Inception model expects input in range [-1, 1]
        images = (images / 2.0) + 0.5

        # Get Inception model's predictions
        with torch.no_grad():
            images = images.to(self.device)

            logits = self.inception(images)

        # Calculate softmax
        preds = F.softmax(logits, dim=1)

        # Calculate the Inception Score
        kl_divergence = preds * (torch.log(preds) - torch.log(preds.mean(dim=0)))

        score = kl_divergence.sum(dim=1).mean().item()

        return score


class FrechetInceptionDistance(Metric):
    def __init__(self,num_samples: int = 10,latent_dim: int = 64, use_cuda: bool = True):
        super().__init__("Frechet Inception Distance",num_samples=num_samples,latent_dim=latent_dim, use_cuda=use_cuda)

    def calculate_activation_statistics(self, images: torch.Tensor, conditions: torch.Tensor = None):
        """
        Calculate the mean and covariance of the activations of the Inception model.

        Args:
            images (torch.Tensor): Batch of images.
            conditions (torch.Tensor): Batch of conditions (if applicable).

        Returns:
            torch.Tensor: Mean of the activations.
            torch.Tensor: Covariance matrix of the activations.
        """
        # Resize images to Inception model's input size (299x299)
        images = interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Inception model expects input in range [-1, 1]
        images = (images / 2.0) + 0.5

        # Get Inception model's predictions (activations)
        with torch.no_grad():
            images = images.to(self.device)
            activations = self.inception(images)

        # Flatten the activations to obtain feature vectors
        activations = activations.view(activations.size(0), -1)

        # If conditions are provided, concatenate them to the feature vectors
        if conditions is not None:
            conditions = conditions.unsqueeze(1).repeat(1, activations.size(1))
            activations = torch.cat([activations, conditions], dim=1)

        # Calculate mean and covariance of the activations
        mean = torch.mean(activations, dim=0)
        cov_matrix = self.torch_cov(activations, rowvar=False)

        return mean, cov_matrix

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate the Frechet distance between two multivariate Gaussian distributions.

        Args:
            mu1 (torch.Tensor): Mean of the first distribution.
            sigma1 (torch.Tensor): Covariance matrix of the first distribution.
            mu2 (torch.Tensor): Mean of the second distribution.
            sigma2 (torch.Tensor): Covariance matrix of the second distribution.

        Returns:
            torch.Tensor: Frechet distance.
        """
        # Calculate squared L2 norm between means
        diff_mean = mu1 - mu2
        term1 = torch.sum(diff_mean ** 2)

        # Calculate matrix trace term
        term2 = torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(torch.clamp(torch.matmul(sigma1, sigma2), min=0)))

        # Sum of the squared L2 norm and matrix trace
        frechet_distance = term1 + term2

        # Ensure non-negative value
        frechet_distance = torch.abs(frechet_distance)

        return frechet_distance

    def batch_compute(self, inp: torch.Tensor, vae: torch.nn.Module, conditions: torch.nn.Module = None):
        assert len(inp) == 2, "FID computation requires two sets of images (real and generated)."
        real_images, labels = inp
        real_images = real_images.to(self.device)


        random_latents = torch.randn(self.num_samples, self.latent_dim, device=self.device)
        vae = vae.to(self.device)
        conditions = conditions.to(self.device)

        with torch.no_grad():
            # Pass conditions to the decoder during image generation
            generated_images = vae.decoder_module(random_latents, conditions)

        generated_images = torch.cat([generated_images] * 3, dim=1)
        real_images = torch.cat([real_images] * 3, dim=1)

        # Calculate statistics for real images
        real_mean, real_cov = self.calculate_activation_statistics(real_images, conditions)

        # Calculate statistics for generated images
        generated_mean, generated_cov = self.calculate_activation_statistics(generated_images, conditions)

        # Calculate Frechet distance
        fid = self.calculate_frechet_distance(real_mean, real_cov, generated_mean, generated_cov)
        return fid

    @staticmethod
    def torch_cov(m, rowvar=False):
        """
        Estimate the covariance matrix of a tensor.

        Args:
            m (torch.Tensor): Input tensor.
            rowvar (bool): Whether the input represents rows (True) or columns (False).

        Returns:
            torch.Tensor: Covariance matrix.
        """
        if rowvar:
            m = m.t()
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        return fact * m.matmul(mt)
