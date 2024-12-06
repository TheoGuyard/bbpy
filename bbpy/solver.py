"""Base class for optimization solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from numpy.typing import ArrayLike

from .problem import Problem


class Status(Enum):
    """Optimization solver status."""

    RUNNING = "running"
    OPTIMAL = "optimal"
    RELATIVE_OPTIMAL = "relative_optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ITER_LIMIT = "iter_limit"
    MEMORY_LIMIT = "memory_limit"
    ERROR = "error"


@dataclass
class Result:
    """Optimization solver result.

    Attributes
    ----------
    status : Status
        The solver status.
    solution : ArrayLike
        The solution.
    objective_value : float
        The objective value.
    solve_time : float
        The solve time.
    iterations : int
        The number of iterations.
    trace : list
        The solver trace.
    """

    status: Status
    solution: ArrayLike
    objective_value: float
    solve_time: float
    iterations: int
    trace: list

    def __repr__(self):
        s = "\n".join(
            [
                "Result",
                f"  status: {self.status}",
                f"  value : {self.objective_value}",
                f"  time  : {self.solve_time}",
                f"  iter  : {self.iterations}",
            ]
        )
        return s


class Solver(ABC):
    """Base class for optimization solvers."""

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """
        Solve an optimization problem.

        Parameters
        ----------
        problem : Problem
            The optimization problem to solve.

        Returns
        -------
        Result
            The solver result.
        """
        pass
