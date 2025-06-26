import torch
from torch.distributions import MultivariateNormal, InverseGamma
from torch.distributions.kl import kl_divergence
from .base_model import BaseModel

class ConjugateSpatialLM(BaseModel):
    def __init__(self, V_beta: list, mu_beta: list, a_sigma: float, b_sigma: float, phi: float, delta2: float):
        """
        Bayesian Linear Geostatistical Model with conjugate priors.

        Args:
            V_beta: Covariance of the prior for beta (V_beta).
            mu_beta: Mean of the prior for beta (mu_beta).
            a_sigma: Shape parameter of the inverse-gamma prior on sigma^2.
            b_sigma: Scale parameter of the inverse-gamma prior on sigma^2.
            phi: Spatial decay parameter for the correlation matrix (phi).
            delta2: Noise-to-spatial variance ratio (delta^2).
        """
        self.V_beta = torch.tensor(V_beta, dtype=torch.float32)
        self.mu_beta = torch.tensor(mu_beta, dtype=torch.float32)
        self.a_sigma = a_sigma
        self.b_sigma = b_sigma
        self.phi = phi
        self.delta2 = delta2

        self.posterior = None

    def compute_posterior(self, data_dict: dict) -> tuple:
        """
        Compute the posterior distribution of beta and sigma^2.

        Expects data_dict to have:
          - "locations": shape (n, 2)
          - "X": shape (n, p_cov)  (no intercept included)
          - "y": shape (n,)

        We add an intercept column of ones inside this method.
        """
        locations = data_dict["locations"]  # (n, 2)
        X_raw = data_dict["X"]             # (n, p_cov), no intercept
        y = data_dict["y"]                 # (n,)

        n, p_cov = X_raw.shape

        # Add intercept column of 1's
        ones_col = torch.ones((n, 1), dtype=X_raw.dtype, device=X_raw.device)
        X = torch.cat([ones_col, X_raw], dim=1)  # (n, 1 + p_cov)

        # Compute spatial correlation matrix
        dists = torch.cdist(locations, locations, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        R_phi = torch.exp(-self.phi * dists)

        # Covariance matrix for y
        Vy = R_phi + self.delta2 * torch.eye(n, device=X.device)

        L = torch.linalg.cholesky(Vy)
        Vy_inv = torch.cholesky_inverse(L)

        # Invert prior covariance
        V_beta_inv = torch.linalg.inv(self.V_beta)

        # Posterior parameters
        M_inv = V_beta_inv + X.T @ Vy_inv @ X
        L2 = torch.linalg.cholesky(M_inv)
        M = torch.cholesky_inverse(L2)

        m = M @ (V_beta_inv @ self.mu_beta + X.T @ Vy_inv @ y)

        a_sigma_post = self.a_sigma + n / 2
        b_sigma_post = self.b_sigma + 0.5 * (
            y @ Vy_inv @ y
            + self.mu_beta.T @ V_beta_inv @ self.mu_beta
            - m.T @ M_inv @ m
        )

        posterior_sigma2 = InverseGamma(a_sigma_post, b_sigma_post)
        posterior_beta_given_sigma2 = lambda sigma2: MultivariateNormal(m, sigma2 * M)

        self.posterior = (posterior_beta_given_sigma2, posterior_sigma2)
        return self.posterior

    def generate_synthetic_data(
        self,
        num_locations: int,
        num_covariates: int,
        true_beta: torch.Tensor,
        true_sigma2: float
    ) -> dict:
        """
        Generate synthetic data for the model, but X will NOT include the intercept column.
        The intercept must be added inside compute_posterior.

        Args:
            num_locations: Number of spatial locations (n).
            num_covariates: Number of covariates (p_cov, excluding intercept).
            true_beta: True regression coefficients (including intercept).
                      shape => (1 + p_cov,) if you want an intercept in the true model
            true_sigma2: True variance parameter.

        Returns:
            A dictionary with:
              - "locations": shape (n,2)
              - "X": shape (n, p_cov) (no intercept)
              - "y": shape (n,)
        """
        # Generate random spatial locations
        locations = (
            torch.randn((num_locations, 2), dtype=torch.float32) * 5.0
            + torch.tensor([40.0, -75.0], dtype=torch.float32)
        )

        # Spatial correlation
        dists = torch.cdist(locations, locations, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        dists.fill_diagonal_(0.0)
        R_phi = torch.exp(-self.phi * dists)

        # Covariance for y
        Vy = true_sigma2 * (R_phi + self.delta2 * torch.eye(num_locations))

        # Check if Vy is symmetric
        if not torch.allclose(Vy, Vy.T):
            raise ValueError("Vy matrix is not symmetric.")

        # Generate covariates (without intercept)
        X_raw = torch.randn((num_locations, num_covariates), dtype=torch.float32)

        # Build design matrix with intercept to produce y
        # true_beta should have length = 1 + p_cov
        # The first entry is the intercept
        intercept = true_beta[0]
        betas = true_beta[1:]  # shape (p_cov,)

        # mean = intercept + X_raw * betas
        # More explicitly:
        #    mean(i) = intercept + sum_j( X_raw(i,j) * betas[j] )
        mean = intercept + (X_raw @ betas)

        response_distribution = MultivariateNormal(mean, Vy)
        y = response_distribution.sample()

        return {
            "locations": locations,  # (n, 2)
            "X": X_raw,              # (n, p_cov)
            "y": y                   # (n,)
        }

    def define_adversarial_posterior(self, params: dict):
        beta_mean = torch.tensor(params.get("beta_mean", self.mu_beta), dtype=torch.float32)
        beta_cov = torch.tensor(params.get("beta_cov", self.V_beta), dtype=torch.float32)
        sigma2_shape = params.get("sigma2_shape", self.a_sigma)
        sigma2_scale = params.get("sigma2_scale", self.b_sigma)

        adversarial_sigma2 = InverseGamma(sigma2_shape, sigma2_scale)
        adversarial_beta_given_sigma2 = lambda sigma2: MultivariateNormal(beta_mean, sigma2 * beta_cov)
        return adversarial_beta_given_sigma2, adversarial_sigma2

    def kl_divergence(self, target_posterior: tuple, objective_fn: str = "exclusive") -> torch.Tensor:
        if self.posterior is None:
            raise ValueError("Posterior has not been computed yet.")
        posterior_beta, posterior_sigma2 = self.posterior
        target_beta, target_sigma2 = target_posterior

        if objective_fn == "exclusive":
            expected_sigma2 = posterior_sigma2.mean
            kl_sigma2 = kl_divergence(posterior_sigma2, target_sigma2)
            kl_beta = kl_divergence(posterior_beta(expected_sigma2), target_beta(expected_sigma2))
        elif objective_fn == "inclusive":
            expected_sigma2 = target_sigma2.mean
            kl_sigma2 = kl_divergence(target_sigma2, posterior_sigma2)
            kl_beta = kl_divergence(target_beta(expected_sigma2), posterior_beta(expected_sigma2))
        else:
            raise ValueError("Invalid objective_fn. Use 'exclusive' or 'inclusive'.")

        return kl_sigma2 + kl_beta
