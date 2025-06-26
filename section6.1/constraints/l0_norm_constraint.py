# constraints/l0_norm_constraint.py
import torch
from .base_constraint import Constraint

class L0NormConstraint(Constraint):
    def __init__(self, max_total_changes, max_row_changes):
        """
        Initialize the L0 norm constraint with global and uniform row-wise limits.

        Args:
            max_total_changes: Maximum number of total changes allowed (\( b_1 \)).
            max_row_changes: Maximum number of changes allowed per row (\( b_{2n} \)).
        """
        self.max_total_changes = max_total_changes
        self.max_row_changes = max_row_changes

    def __call__(self, X, X_prime):
        """
        Enforce the L0 constraint by applying uniform row-wise and global limits.

        Args:
            X: Current matrix (\( n \times p \)).
            X_prime: Reference matrix (\( n \times p \)).

        Returns:
            Projected matrix (\( \mathbf{X} \)).
        """
        # Make a copy of the data to project
        X_projected = X.clone()

        # Step 1: Apply row-wise constraint
        row_changes = torch.zeros(X.size(0), dtype=torch.int32)  # Track row-wise changes
        for row_idx in range(X.size(0)):
            row_diff = (X[row_idx] - X_prime[row_idx]).abs()
            num_diff = row_diff.nonzero().numel()  # Count differences

            if num_diff > self.max_row_changes:
                _, indices = torch.topk(row_diff, self.max_row_changes)  # Top-k largest differences
                row_mask = torch.zeros_like(row_diff, dtype=torch.bool)
                row_mask[indices] = True  # Keep only the largest allowed differences
                X_projected[row_idx] = X_prime[row_idx]
                X_projected[row_idx][row_mask] = X[row_idx][row_mask]
                row_changes[row_idx] = self.max_row_changes
            else:
                row_changes[row_idx] = num_diff

        # Step 2: Apply global constraint
        total_changes = row_changes.sum().item()
        if total_changes > self.max_total_changes:
            # Compute global differences after row-wise constraint
            diff = (X_projected - X_prime).abs()
            _, indices = torch.topk(diff.flatten(), self.max_total_changes)
            global_mask = torch.zeros_like(diff, dtype=torch.bool)
            global_mask.view(-1)[indices] = True

            # Reapply projection to enforce global constraint
            X_projected = X_prime.clone()
            X_projected[global_mask] = X[global_mask]

        return X_projected
