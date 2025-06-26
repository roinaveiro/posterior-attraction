# constraints/combined_constraint.py
import torch
from .base_constraint import Constraint

class CombinedConstraintL1(Constraint):
    def __init__(self, max_row_l1_norm, max_global_l1_norm, max_element_change):
        """
        Initialize the combined constraint for row-wise L1, global L1, and box constraints.

        Args:
            max_row_l1_norm: Maximum allowed L1 norm per row (\( b_{2i} \)).
            max_global_l1_norm: Maximum allowed global L1 norm (\( b_1 \)).
            max_element_change: Maximum allowed change per element (\( b_{3ij} \)).
        """
        self.max_row_l1_norm = max_row_l1_norm
        self.max_global_l1_norm = max_global_l1_norm
        self.max_element_change = max_element_change

    def __call__(self, Z, X_prime):
        """
        Project the matrix Z onto the feasible set defined by the constraints.

        Args:
            Z: Current matrix (\( n \times p \)), the target matrix to project.
            X_prime: Reference matrix (\( n \times p \)).

        Returns:
            Projected matrix (\( X \)).
        """
        # Step 1: Apply box constraints
        X = torch.clamp(Z, X_prime - self.max_element_change, X_prime + self.max_element_change)

        # Step 2: Apply row-wise L1 constraints
        for i in range(X.shape[0]):
            row_diff = X[i] - X_prime[i]
            row_l1_norm = torch.sum(torch.abs(row_diff))
            if row_l1_norm > self.max_row_l1_norm:
                scaling_factor = self.max_row_l1_norm / row_l1_norm
                X[i] = X_prime[i] + scaling_factor * row_diff

        # Step 3: Apply global L1 constraint
        global_diff = X - X_prime
        global_l1_norm = torch.sum(torch.abs(global_diff))
        if global_l1_norm > self.max_global_l1_norm:
            scaling_factor = self.max_global_l1_norm / global_l1_norm
            X = X_prime + scaling_factor * global_diff

        return X
