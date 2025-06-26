# constraints/l1_norm_constraint.py
import torch
from .base_constraint import Constraint

class L1NormConstraint(Constraint):
    def __init__(self, max_element_changes):
        """
        Initialize the L1 norm constraint.

        Args:
            max_element_changes: Maximum absolute difference for each element (\( b_{3np} \)).
        """
        self.max_element_changes = max_element_changes

    def __call__(self, X, X_prime):
        """
        Enforce the L1 constraint by clamping the differences.

        Args:
            X: Current matrix (\( n \times p \)).
            X_prime: Reference matrix (\( n \times p \)).

        Returns:
            Projected matrix (\( \mathbf{X} \)).
        """
        diff = X - X_prime
        diff = torch.clamp(diff, -self.max_element_changes, self.max_element_changes)
        return X_prime + diff
