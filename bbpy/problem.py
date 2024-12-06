"""Base class for optimization problems."""

from abc import ABC, abstractmethod


class Problem(ABC):
    """Base class for optimization problems."""

    @abstractmethod
    def value(self, x) -> float:
        """Value of the objective function at a given point x.

        Parameters
        ----------
        x : Any
            Point to evaluate.

        Returns
        -------
        value : float
            Value of the objective function at x.
        """
        pass

    @abstractmethod
    def is_feasible(self, x, feas_tol: float = 1e-12) -> bool:
        """Check if a given point x is feasible for the problem.

        Parameters
        ----------
        x : Any
            Point to check.
        feas_tol : float, default=1e-12
            Tolerance for the feasibility constraints.

        Returns
        -------
        feasible : bool
            Whether x is feasible for the problem.
        """
        pass
