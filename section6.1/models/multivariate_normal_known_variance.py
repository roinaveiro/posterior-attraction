# models/multivariate_normal_known_variance.py
import torch
from torch.distributions import MultivariateNormal
from .base_model import BaseModel
from torch.distributions.kl import kl_divergence

class MultivariateNormalKnownVarianceModel(BaseModel):
    def __init__(self, prior_mean: list, prior_cov: list, known_cov: list):
        """
        Multivariate Normal model with a Gaussian prior on the mean and known covariance.

        Args:
            prior_mean: Mean of the prior (\( \mu_0 \)).
            prior_cov: Covariance of the prior (\( \Sigma_0 \)).
            known_cov: Known covariance of the likelihood (\( \Sigma \)).
        """
        self.known_cov = torch.tensor(known_cov, dtype=torch.float32)
        prior = MultivariateNormal(
            torch.tensor(prior_mean, dtype=torch.float32),
            torch.tensor(prior_cov, dtype=torch.float32)
        )
        super().__init__(prior)

    def generate_synthetic_data(self, params: dict, size: int) -> torch.Tensor:
        """
        Generate synthetic data for the model.

        Args:
            params: Dictionary with distribution parameters (e.g., mean).
            size: Number of samples to generate.

        Returns:
            Tensor of generated data.
        """

        # Set seed
        torch.manual_seed(0)

        mean = torch.tensor(params["mean"], dtype=torch.float32)
        distribution = MultivariateNormal(mean, self.known_cov)

        data_dict = {
            "X": distribution.sample((size,))
        }

        return data_dict
    
    def compute_posterior(self, data_dict: dict) -> MultivariateNormal:
        """
        Compute the posterior distribution of the mean with known covariance.

        Args:
            data_dict: Dictionary containing the data.

        Returns:
            Posterior distribution.
        """
        data = data_dict["X"]
        n, d = data.shape
        sample_mean = data.mean(dim=0)
        prior_mean = self.prior.mean
        prior_cov = self.prior.covariance_matrix
        known_cov = self.known_cov

        # Posterior covariance
        posterior_cov = torch.inverse(
            torch.inverse(prior_cov) + n * torch.inverse(known_cov)
        )

        # Posterior mean
        posterior_mean = posterior_cov @ (
            torch.inverse(prior_cov) @ prior_mean + n * torch.inverse(known_cov) @ sample_mean
        )

        self.posterior = MultivariateNormal(posterior_mean, posterior_cov)
        return self.posterior

            
    def define_adversarial_posterior(self, target_params: dict, n: int) -> MultivariateNormal:
        """
        Define the adversarial (target) posterior with optional covariance.

        Args:
            target_params: Dictionary containing the mean and optionally covariance for the target posterior.
            n: Number of data points.

        Returns:
            MultivariateNormal distribution representing the adversarial posterior.
        """
        mean = torch.tensor(target_params["mean"], dtype=torch.float32)

        if "covariance" in target_params:
            # Use the specified covariance
            covariance = torch.tensor(target_params["covariance"], dtype=torch.float32)
        else:
            # Compute the posterior covariance
            prior_cov = self.prior.covariance_matrix
            known_cov = self.known_cov
            covariance = torch.inverse(
                torch.inverse(prior_cov) + n * torch.inverse(known_cov)
            )

        return MultivariateNormal(mean, covariance)

    def kl_divergence(self, target_posterior: MultivariateNormal, 
                        objective_fn: str = "exclusive") -> torch.Tensor:
        """
        Compute the KL divergence between the computed posterior and the target posterior.

        Args:
            target_posterior: The target distribution (\( \pi_A(\theta) \)).
            objective_fn: The KL divergence direction:
                - "exclusive" for \( D_{KL}(\text{posterior} || \text{target}) \)
                - "inclusive" for \( D_{KL}(\text{target} || \text{posterior}) \)

        Returns:
            torch.Tensor: The KL divergence.
        """
        if self.posterior is None:
            raise ValueError("Posterior has not been computed yet.")
        
        if objective_fn == "exclusive":
            return kl_divergence(self.posterior, target_posterior)
        elif objective_fn == "inclusive":
            return kl_divergence(target_posterior, self.posterior)
        else:
            raise ValueError("Invalid objective_fn. Use 'exclusive' or 'inclusive'.")


