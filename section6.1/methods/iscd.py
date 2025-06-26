import itertools
import torch
from torch.autograd.functional import vjp
from methods.base_method import BaseMethod

class ISCD(BaseMethod):
    """
    Integer-Steps Coordinate Descent with:
      - A dictionary of data (data_dict)
      - A matching dictionary of constraints (constraints)
        * constraints[k]["max_element_change"] -> per-element limit (same shape as data_dict[k])
        * constraints[k]["max_changes"] -> an integer limit on how many elements
                                           of data_dict[k] can ever be changed
        * constraints["X"]["max_row_changes"] -> 1D tensor, row constraints (only applies to "X")
      - Oscillation handling by coordinate masking
      - Optional second-order HVP
    """

    def __init__(
        self,
        model,
        target_posterior,
        kl_direction,
        epsilon,
        constraints,  # dictionary of per-key constraints
        max_oscillations=3,
        second_order=False,
        verbose=False
    ):
        """
        Args:
            model: Your model (BaseModel child) with `compute_posterior(data_dict)` method.
            target_posterior: Target distribution (for KL).
            kl_direction: String indicating the desired KL divergence (exclusive or inclusive).
            epsilon: Step size for each update (integer step).
            constraints: Dictionary of dictionaries. For each key K in data_dict:
                constraints[K]["max_element_change"]: Tensor same shape as data_dict[K].
                constraints[K]["max_changes"]: int, how many total elements in data_dict[K] can be changed.
                If K == "X", also constraints["X"]["max_row_changes"]: 1D tensor of length X.shape[0].
            max_oscillations: # of sign flips allowed before we mask a coordinate out.
            second_order: If True, approximate second-order local improvement using Hessian diag.
            verbose: If True, prints out iteration info.
        """
        super().__init__(model, target_posterior, kl_direction)
        self.epsilon = epsilon
        self.constraints = constraints
        self.max_oscillations = max_oscillations
        self.second_order = second_order
        self.verbose = verbose

        # Tracking how many changes we have made to each key
        # Key -> int (count of elements that have ever been changed in that tensor)
        self.key_total_changes = {k: 0 for k in constraints.keys()}

        # For each coordinate, how much we've changed from the initial reference
        self.total_perturbation = {}
        self.modified = {}
        self.oscillation_counters = {}
        self.last_signs = {}
        # If the "X" key has row constraints, track changes per row
        self.row_changes = {}

    def _compute_kl_loss(self, data_dict):
        self.model.compute_posterior(data_dict)
        return self.model.kl_divergence(self.target_posterior, self.kl_direction)

    def _init_trackers(self, data_dict):
        """
        Initialize bookkeeping structures to track total perturbation, 
        modifications, sign flips (oscillations), etc.
        """
        for key, tensor in data_dict.items():
            if tensor is None or not tensor.requires_grad:
                continue

            shape = tensor.shape
            self.total_perturbation[key] = torch.zeros_like(tensor)
            self.modified[key] = torch.zeros(shape, dtype=torch.bool)
            self.oscillation_counters[key] = torch.zeros(shape, dtype=torch.int)
            self.last_signs[key] = torch.zeros_like(tensor, dtype=torch.float)

            # If there's a row constraint and the key is "X"
            if key == "X" and "max_row_changes" in self.constraints.get("X", {}):
                # row_changes: track how many elements changed in each row
                self.row_changes[key] = torch.zeros(shape[0], dtype=torch.int)

    def _compute_gradient(self, data_dict):
        """Compute gradient of KL wrt each requires_grad tensor in data_dict."""
        # Zero grads
        for t in data_dict.values():
            if t is not None and t.requires_grad and t.grad is not None:
                t.grad.zero_()

        # Forward
        kl_loss = self._compute_kl_loss(data_dict)
        # Backward
        kl_loss.backward()

        # Gather gradients
        gradient_dict = {}
        for key, tensor in data_dict.items():
            if tensor is not None and tensor.requires_grad:
                gradient_dict[key] = tensor.grad.clone()
        return kl_loss, gradient_dict

    def _compute_hessian_diag(self, data_dict):
        """
        Naive coordinate-wise Hessian diag via finite vjp calls.
        If second_order=False, we skip this (performance reasons).
        """
        hess_dict = {}

        def kl_loss_fn(dict_in):
            self.model.compute_posterior(dict_in)
            return self.model.kl_divergence(self.target_posterior, self.objective_fn)

        for key, tensor in data_dict.items():
            if tensor is None or not tensor.requires_grad:
                continue

            hessian_diag = torch.zeros_like(tensor)
            flat_tensor = tensor.view(-1)
            num_coords = flat_tensor.numel()

            for idx in range(num_coords):
                unit_vec = torch.zeros_like(flat_tensor)
                unit_vec[idx] = 1.0

                def grad_of_kl(x):
                    temp_dict = dict(data_dict)
                    temp_dict[key] = x
                    out = kl_loss_fn(temp_dict)
                    return torch.autograd.grad(out, x, create_graph=True)[0]

                _, hvp = vjp(grad_of_kl, tensor, v=unit_vec.view_as(tensor))
                hessian_diag.view(-1)[idx] = hvp.view(-1)[idx]

            hess_dict[key] = hessian_diag
        return hess_dict

    def _mask_coordinate(self, key, idx):
        """
        Mask out a coordinate from further updates (e.g. due to oscillation).
        We can do this by setting total_perturbation to max_element_change 
        so it always fails feasibility. Also mark as modified.
        """
        max_elem_change_tensor = self.constraints[key]["max_element_change"]
    

        if key in self.total_perturbation:
            self.total_perturbation[key][idx] = max_elem_change_tensor[idx].item()
        if key in self.modified:
            self.modified[key][idx] = True

    def _feasible_coordinate(self, key, idx):
        """
        Check if coordinate (key, idx) can still be updated:
          - If we've already changed it, do we exceed element-wise max_element_change?
          - If not changed yet, do we exceed the row constraints (if X),
            or the per-key max_changes?
        """
        # Skip if that key isn't being tracked
        if key not in self.modified:
            return False
        
        # Already changed => check if we've maxed out how far we can move
        if self.modified[key][idx]:
            # total_perturbation vs. constraints[key]["max_element_change"]
            m = self.constraints[key]["max_element_change"][idx].item()
            if abs(self.total_perturbation[key][idx]) >= m:
                return False
        else:
            # Also check key-level max_changes
            if self.key_total_changes[key] >= (self.constraints[key]["max_changes"]):
                return False
            
            # If not changed yet => check row constraint if it's X
            if key == "X" and "max_row_changes" in self.constraints["X"]:
                row_idx = idx[0]
                if self.row_changes[key][row_idx] >= self.constraints["X"]["max_row_changes"][row_idx]:
                    return False


        return True

    def minimize_kl(self, reference_data_dict, **kwargs):
        """
        Main loop of integer-step coordinate descent with per-key constraints.
        
        Args:
            reference_data_dict: Dictionary of reference Tensors. 
                                 Typically your "starting point."
        Returns:
            data_dict: The final (optimized) dictionary respecting constraints.
        """
        # 1) Clone dictionary so we don't overwrite user data
        data_dict = {}
        for key, val in reference_data_dict.items():
            if val is not None:
                data_dict[key] = val.clone().detach().requires_grad_(val.requires_grad)
            else:
                data_dict[key] = None

        # 2) Initialize trackers
        self._init_trackers(data_dict)

        # 3) Optimization loop
        while True:
            # (a) Compute gradient & KL
            kl_loss, gradient_dict = self._compute_gradient(data_dict)

            # (b) Hess diag if needed
            hess_dict = {}
            if self.second_order:
                hess_dict = self._compute_hessian_diag(data_dict)

            # (c) Find best coordinate to update (greedy)
            best_score = float('inf')
            best_key = None
            best_idx = None
            best_grad_val = 0.0

            for key, tensor in data_dict.items():
                if tensor is None or not tensor.requires_grad:
                    continue

                g = gradient_dict[key]
                if key in hess_dict:
                    hdiag = hess_dict[key]
                else:
                    hdiag = None

                # Iterate over all coordinates
                for idx in itertools.product(*(range(s) for s in tensor.shape)):
                    if not self._feasible_coordinate(key, idx):
                        continue

                    grad_val = g[idx].item()
                    if self.second_order and hdiag is not None:
                        h_val = hdiag[idx].item()
                        # second-order local improvement estimate
                        local_score = -abs(grad_val)*self.epsilon + 0.5*h_val*(self.epsilon**2)
                    else:
                        # first-order
                        local_score = -abs(grad_val)*self.epsilon

                    # We want the "best improvement" (most negative local_score)
                    if local_score < 0 and local_score < best_score:
                        best_score = local_score
                        best_key = key
                        best_idx = idx
                        best_grad_val = grad_val

            # (d) If no feasible coordinate found => stop
            if best_score == float('inf'):
                if self.verbose:
                    print("No more feasible coordinates. Stopping.")
                break

            # (e) Determine sign & handle oscillation
            perturbation = -self.epsilon * torch.sign(torch.tensor(best_grad_val))
            old_sign = self.last_signs[best_key][best_idx].item()
            new_sign = perturbation.sign().item()

            # Check for sign flip
            if old_sign == -new_sign and old_sign != 0:
                self.oscillation_counters[best_key][best_idx] += 1
            else:
                self.oscillation_counters[best_key][best_idx] = 0

            if self.oscillation_counters[best_key][best_idx] >= self.max_oscillations:
                # Mask out if too many flips
                self._mask_coordinate(best_key, best_idx)
                if self.verbose:
                    print(f"[Oscillation] Masking out {best_key}[{best_idx}].")
                continue

            # (f) Clip by element-wise max_element_change
            max_elem_change_tensor = self.constraints[best_key]["max_element_change"]
            current_val = self.total_perturbation[best_key][best_idx]
            remain = max_elem_change_tensor[best_idx].item() - abs(current_val)
            # The actual step we can still take
            actual_perturb = min(remain, abs(perturbation.item()))
            actual_perturb = actual_perturb * torch.sign(perturbation)

            # (g) Apply update
            data_dict[best_key].data[best_idx] += actual_perturb
            self.total_perturbation[best_key][best_idx] += actual_perturb

            # If first time changing this coordinate, update "modified" + key-level count + row-level count
            if not self.modified[best_key][best_idx]:
                self.modified[best_key][best_idx] = True
                self.key_total_changes[best_key] += 1

                # If row constraints apply (key == "X")
                if best_key == "X" and "max_row_changes" in self.constraints["X"]:
                    row_idx = best_idx[0]
                    self.row_changes[best_key][row_idx] += 1

            self.last_signs[best_key][best_idx] = new_sign

            if self.verbose:
                print(f"Updated {best_key}[{best_idx}] by {actual_perturb:.4f}, "
                      f"score={best_score:.6f}, KL={kl_loss.item():.6f}, "
                      f"total_changes_for_{best_key}={self.key_total_changes[best_key]}, "
                      f"osc_count={self.oscillation_counters[best_key][best_idx]}")

            # (h) Check if we've hit max_changes for this key
            if self.key_total_changes[best_key] >= self.constraints[best_key]["max_changes"]:
                # That means we can no longer change ANY new element in best_key
                # We can still potentially change other keys, though.
                # So we can mark all *unmodified* coords in best_key as infeasible
                # or simply rely on `_feasible_coordinate` logic (see below).
                pass

        return data_dict
