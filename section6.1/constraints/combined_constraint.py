# constraints/combined_constraint.py
import torch
from .base_constraint import Constraint


class CombinedConstraint(Constraint):
    def __init__(self, max_row_changes, max_total_changes, max_element_change):
        """
        Combined constraint for row-wise, global sparsity, and element-wise box constraints.

        Args:
            max_row_changes: 1D tensor of length n, specifying the maximum number
                             of changes allowed per row i (instead of a single integer).
            max_total_changes: Maximum number of total changes allowed (c).
            max_element_change: 2D tensor of shape (n, p), specifying the maximum allowed
                                change for each element (i,j).
        """
        super().__init__()
        self.max_row_changes = max_row_changes  # shape: (n,)
        self.max_total_changes = max_total_changes
        self.max_element_change = max_element_change  # shape: (n, p)

    def __call__(self, Z, X_prime):
        """
        Project the matrix Z onto the feasible set defined by:
          - row-wise sparsity: at most max_row_changes[i] changes per row i
          - total sparsity: at most max_total_changes changes across all elements
          - element-wise box constraint: each element Z[i, j] is clipped to
            [X_prime[i, j] - max_element_change[i, j], X_prime[i, j] + max_element_change[i, j]]

        Args:
            Z: Current matrix (n x p), the target to project.
            X_prime: Reference matrix (n x p).

        Returns:
            A projected matrix X that respects the constraints.
        """
        # Basic shape checks (optional, but safer)
        n, p = Z.shape
        if self.max_row_changes.shape[0] != n:
            raise ValueError("max_row_changes must have the same number of rows as Z.")
        if self.max_element_change.shape != (n, p):
            raise ValueError("max_element_change must match the shape of Z.")

        # 1) Initialize X as a copy of X'.
        X = X_prime.clone()

        # 2) Element-wise box clipping: [-max_element_change, +max_element_change] around X_prime
        lower_bound = X_prime - self.max_element_change
        upper_bound = X_prime + self.max_element_change
        clipped_Z = torch.clamp(Z, lower_bound, upper_bound)

        # 3) Compute the potential error reduction for each element (like a “greedy” measure).
        delta = (X_prime - Z)**2 - (clipped_Z - Z)**2

        # 4) Flatten indices & delta for sorting.
        indices = [(i, j) for i in range(n) for j in range(p)]
        delta_flat = delta.flatten()  # shape: (n*p,)

        # 5) Sort indices by decreasing delta (bigger delta => bigger “gain” from changing).
        sorted_indices = torch.argsort(-delta_flat).tolist()

        # 6) Initialize counters for row-wise and global changes
        row_changes = torch.zeros(n, dtype=torch.int32)  # track # of changes in each row
        global_changes = 0

        # 7) Greedy application of changes within constraints
        for flat_idx in sorted_indices:
            i, j = divmod(flat_idx, p)  # map flattened idx back to matrix coords

            # Global sparsity constraint
            if global_changes >= self.max_total_changes:
                break

            # Row-wise constraint
            if row_changes[i] >= self.max_row_changes[i].item():
                continue

            # Apply the change
            X[i, j] = clipped_Z[i, j]
            row_changes[i] += 1
            global_changes += 1

        return X