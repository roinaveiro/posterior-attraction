class BaseModel:
    def __init__(self, prior):
        """
        Base class for models.

        Args:
            prior: A PyTorch distribution representing the prior.
        """
        self.prior = prior
        self.posterior = None

    def compute_posterior(self, data):
        """
        Computes the posterior distribution given the data.
        Subclasses must implement this.

        Args:
            data: The dataset X.

        Returns:
            A PyTorch distribution representing the posterior.
        """
        raise NotImplementedError("Subclasses must implement this method.")

