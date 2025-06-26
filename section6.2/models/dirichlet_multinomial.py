import torch
from torch.distributions import Dirichlet, Multinomial
from torch.distributions.kl import kl_divergence
from .base_model import BaseModel


class DirichletMultinomialModel(BaseModel):
    """
    A Dirichlet–Multinomial Bayesian model:

      p ~ Dirichlet(alpha0)
      X | p ~ Multinomial(n, p)

    Posterior update:
      alpha_post = alpha0 + sum_of_observed_counts
    """

    def __init__(self, prior_alpha: list):
        """
        Initialize with a Dirichlet prior distribution.

        Args:
            prior_alpha: 1D list or tensor of shape (k,) for the Dirichlet concentration parameters.
        """
        # Convert to torch.Tensor
        prior_alpha_t = torch.tensor(prior_alpha, dtype=torch.float32)

        # Create a Dirichlet prior distribution
        self.prior = Dirichlet(prior_alpha_t)

        # Call the parent constructor to store self.prior in BaseModel
        super().__init__(self.prior)

        # We'll store the computed posterior in self.posterior (analogous to your Normal model).
        self.posterior = None

    def generate_synthetic_data(self, params: dict, size: int) -> dict:
        """
        Generate synthetic data from a Multinomial distribution with "true" probabilities p.

        Args:
            params: Must include:
                - "p": a probability vector of shape (k,)
                - "n": number of trials per Multinomial draw
            size: The number of i.i.d. Multinomial samples you want to generate

        Returns:
            A dict containing:
                - "X": a (size x k) tensor of counts from the Multinomial
        """
        torch.manual_seed(0)  # for reproducibility, if desired

        p_true = torch.tensor(params["p"], dtype=torch.float32)
        n_true = params["n"]

        # Create a Multinomial distribution
        distribution = Multinomial(total_count=n_true, probs=p_true)

        # Sample 'size' times => shape (size, k)
        samples = distribution.sample((size,))  # each row is a count vector

        return {"X": samples}

    def compute_posterior(self, data_dict: dict) -> Dirichlet:
        """
        Given observed data (count vectors), compute the posterior Dirichlet distribution.

        Args:
            data_dict: Must contain "X" of shape (num_samples, k), i.e. multiple Multinomial samples
                       or a single row with aggregated counts.

        Returns:
            A Dirichlet distribution representing the posterior over p.
        """
        X = data_dict["X"]  # shape: (num_samples, k)
        counts_sum = X.sum(dim=0)  # shape: (k,)

        alpha_prior = self.prior.concentration  # shape: (k,)

        # Posterior alpha = alpha_prior + sum_of_counts
        alpha_post = alpha_prior + counts_sum
        self.posterior = Dirichlet(alpha_post)

        return self.posterior

    def define_adversarial_posterior(self, target_params: dict) -> Dirichlet:
        """
        Define a 'target' or 'adversarial' posterior distribution, analogous to define_adversarial_posterior
        in the Normal model.

        Args:
            target_params: dictionary that may contain:
               - "alpha": a 1D list or tensor with Dirichlet concentration parameters
                 If not given, we might default to the prior or something similar.

        Returns:
            A Dirichlet distribution representing the target posterior.
        """
        if "alpha" in target_params:
            alpha_target = torch.tensor(target_params["alpha"], dtype=torch.float32)
        else:
            # If user didn't provide an "alpha", let's default to the prior, for example
            alpha_target = self.prior.concentration.clone()

        return Dirichlet(alpha_target)

    def kl_divergence(self, target_posterior: Dirichlet, objective_fn: str = "exclusive") -> torch.Tensor:
        """
        Compute the KL divergence between the computed posterior and the target posterior:

        - 'exclusive': D_{KL}(posterior || target)
        - 'inclusive': D_{KL}(target || posterior)

        Args:
            target_posterior: a Dirichlet distribution
            objective_fn: "exclusive" or "inclusive"

        Returns:
            A scalar tensor for the KL divergence.
        """
        if self.posterior is None:
            raise ValueError("Posterior has not been computed yet.")

        if objective_fn == "exclusive":
            # D_{KL}( posterior || target )
            return kl_divergence(self.posterior, target_posterior)
        elif objective_fn == "inclusive":
            # D_{KL}( target || posterior )
            return kl_divergence(target_posterior, self.posterior)
        else:
            raise ValueError("Invalid objective_fn. Use 'exclusive' or 'inclusive'.")
