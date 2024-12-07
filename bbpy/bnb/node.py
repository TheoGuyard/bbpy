"""Tree nodes in the Branch-and-Bound algorithm."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Type
from bbpy.problem import Problem


class Region(ABC):
    """Base class for regions in the problem's feasible space."""

    @classmethod
    @abstractmethod
    def root_region(problem: Problem) -> Type["Region"]:
        """
        Get the initial feasible region for the problem.

        Returns
        -------
        Region
            The initial feasible region for the problem.
        """
        pass


class NodeStatus(Enum):
    """Node status."""

    OPEN = "open"
    PRUNED = "pruned"
    FEASIBLE = "feasible"
    PROCESSED = "processed"


class Node:
    """Base class for Branch-and-Bound tree nodes.

    Attributes
    ----------
    status : NodeStatus
        Status of the node.
    region : Region
        Representation of the feasible region for this subproblem.
    level : int
        Depth of the node in the BnB tree.
    x : np.ndarray
        Lower bounding solution for the node.
    lb : float
        Lower bound of the objective function for this node.
    """

    def __init__(
        self,
        region: Region,
        level: int,
        x=None,
        lb: float = -float("inf"),
    ) -> None:
        """
        Initialize a new node.

        Parameters
        ----------
        region : Region
            Representation of the feasible region for this node.
        level : int
            Depth of the node in the BnB tree.
        x : default=None
            Lower bounding solution in the parent node that can be used for
            warm-start purpose.
        lb : float, optional
            Lower bound of the objective function for this node.
        """
        self.status = NodeStatus.OPEN
        self.region = region
        self.level = level
        self.x = x
        self.lb = lb
        self.trace = {}
