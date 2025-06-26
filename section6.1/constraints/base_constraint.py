# constraints/base_constraint.py
class Constraint:
    def __call__(self, X, X_prime):
        """
        Enforce the constraint on the input data.

        Args:
            X: Current matrix (\( n \times p \)).
            X_prime: Reference matrix (\( n \times p \)).

        Returns:
            Projected matrix (\( \mathbf{X} \)).
        """
        raise NotImplementedError("Subclasses must implement this method.")
