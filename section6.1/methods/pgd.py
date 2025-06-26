import torch
from .base_method import BaseMethod

class ProjectedGradientDescent(BaseMethod):
    def __init__(self, 
                 model, 
                 target_posterior, 
                 kl_direction, 
                 lr=0.01, 
                 max_iter=100,
                 tolerance=1e-8, 
                 no_change_steps=50, 
                 optimizer_class=torch.optim.Adam, 
                 verbose=False):
        """
        PGD for minimizing KL divergence with constraints and stopping criteria, 
        but adapted to work with a dictionary of tensors (`data_dict`).

        Args:
            model: Model instance inheriting from BaseModel.
            target_posterior: Target posterior distribution.
            kl_direction: String indicating the desired KL divergence (exclusive or inclusive).
            lr: Learning rate.
            max_iter: Maximum number of iterations.
            tolerance: Minimum change in KL loss for stopping criterion.
            no_change_steps: Number of steps without significant change before stopping.
            optimizer_class: Optimizer class to use (default: Adam).
            verbose: If True, logs the progress during optimization.
        """
        super().__init__(model, target_posterior, kl_direction)
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.no_change_steps = no_change_steps
        self.optimizer_class = optimizer_class
        self.verbose = verbose

    def minimize_kl(self, 
                    data_dict=None, 
                    reference_data_dict=None,
                    constraints=None,
                    random_init=False,
                    init_range=(-1.0, 1.0), 
                    **kwargs):
        """
        Minimize KL divergence with constraints and stopping criteria 
        on a dictionary of tensors.

        Args:
            data_dict: Dictionary { key -> tensor }, the initial data. 
                       If None and `random_init=True`, random initialization 
                       is used (shapes copied from `reference_data_dict`).
            reference_data_dict: Dictionary with the same keys/shapes as data_dict,
                                 used for random initialization or constraints.
            constraints: List of callables that each take (data_dict, reference_data_dict)
                         and modify `data_dict` in-place to satisfy constraints.
            random_init: If True, initialize the entries that require grad 
                         with uniform random values within `init_range`.
            init_range: Tuple specifying the range (low, high) for random initialization.
            kwargs: Additional keyword arguments if needed.

        Returns:
            Optimized dictionary of tensors (`data_dict`).
        """
        if data_dict is None and random_init:
            if reference_data_dict is None:
                raise ValueError("Reference data dict must be provided for random initialization.")
            # Create a dictionary with random initialization 
            # only for those keys whose reference_data requires grad
            data_dict = {}
            for k, ref_tensor in reference_data_dict.items():
                if ref_tensor is not None:
                    # Clone shape; if requires_grad, init randomly; else copy reference
                    if ref_tensor.requires_grad:
                        data_dict[k] = (torch.empty_like(ref_tensor)
                                        .uniform_(*init_range)
                                        .requires_grad_(True))
                    else:
                        # If it doesn't require grad, just copy as a constant
                        data_dict[k] = ref_tensor.clone().detach()
                else:
                    data_dict[k] = None
        elif data_dict is None:
            raise ValueError("Either supply `data_dict` or set `random_init=True` and provide `reference_data_dict`.")

        else:
            # Make sure to detach & requires_grad only if the original says so
            # So we always create a fresh set of Tensors
            for k, v in data_dict.items():
                if v is not None:
                    data_dict[k] = v.clone().detach().requires_grad_(v.requires_grad)

        # Apply initial constraints if any
        if constraints:
            with torch.no_grad():
                for constraint_fn in constraints:
                    data_dict["X"].data = constraint_fn(data_dict["X"], reference_data_dict["X"])

        # Collect params that actually require gradient
        # (i.e., skip dictionary entries with requires_grad=False or is None)
        params_to_optimize = [v for v in data_dict.values()
                              if (v is not None and v.requires_grad)]
        optimizer = self.optimizer_class(params_to_optimize, lr=self.lr)

        # Stopping criteria
        previous_kl_loss = float('inf')
        no_change_counter = 0

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # 1) Forward pass: compute posterior and KL
            self.model.compute_posterior(data_dict)
            kl_loss = self.model.kl_divergence(self.target_posterior, self.kl_direction)

            # 2) Backward pass
            kl_loss.backward()
            optimizer.step()

            # 3) Apply constraints if provided
            if constraints:
                with torch.no_grad():
                    for constraint_fn in constraints:
                        data_dict["X"].data = constraint_fn(data_dict["X"], reference_data_dict["X"]) # Generalize this if needed!

            # 4) Check KL improvement
            current_kl_loss = kl_loss.item()
            kl_change = abs(current_kl_loss - previous_kl_loss)

            # Logging
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: KL Loss = {current_kl_loss:.6f}, "
                      f"Change = {kl_change:.6e}")

            # 5) Early stopping check
            if kl_change < self.tolerance:
                no_change_counter += 1
                if no_change_counter >= self.no_change_steps:
                    if self.verbose:
                        print(f"Stopping criterion met at iteration {iteration}. "
                              f"No significant change for {self.no_change_steps} steps.")
                    break
            else:
                no_change_counter = 0  # reset if there's a significant change

            previous_kl_loss = current_kl_loss

        return data_dict
