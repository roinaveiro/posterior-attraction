class BaseMethod:
    def __init__(self, model, target_posterior, kl_direction):
        """
        Base class for optimization methods.
        
        Args:
            model: A model instance (e.g., NormalNormalModel).
            target_posterior: A PyTorch distribution representing the target posterior.
            kl_direction: String indicating the desired KL divergence (exclusive or inclusive).
        """
        self.model = model
        self.target_posterior = target_posterior
        self.kl_direction = kl_direction

    def minimize_kl(self, data, constraints=None, **kwargs):
        """
        Minimize the given KL divergence objective.
        
        Args:
            data: Initial data to optimize.
            constraints: A callable enforcing constraints on the data.
            kwargs: Additional parameters for the specific optimization method.
        
        Returns:
            Optimized data.
        """
        raise NotImplementedError("Subclasses must implement this method.")
