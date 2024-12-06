"""Bounding methods in the Branch-and-Bound solver."""

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from bbpy.problem import Problem

from .node import Node


class LowerBoundingMethod(ABC):
    """Base class for lower bounding method."""

    def initialize(self, problem: Problem) -> None:
        """Initialize the lower bounding method.

        Parameters
        ----------
        problem : Problem
            The problem to be solved by the Branch-and-Bound solver.
        """
        pass

    @abstractmethod
    def bound(self, node: Node) -> None:
        """Perform the lower bounding operation in the current node. The node
        internal attributes `node.lb` and `node.x` should be updated
        accordingly during the method call.

        Parameters
        ----------
        node : Node
            The current node in the BnB tree.
        """
        pass


class UpperBoundingMethod(ABC):
    """Base class for upper bounding method."""

    def initialize(self, problem: Problem) -> None:
        """Initialize the upper bounding method.

        Parameters
        ----------
        problem : Problem
            The problem to be solved by the Branch-and-Bound solver.
        """
        pass

    @abstractmethod
    def bound(self, node: Node) -> ArrayLike:
        """Perform the upper bounding operation in the current node. The method
        must return the incumbent solution constructed.

        Parameters
        ----------
        node : Node
            The current node in the BnB tree.
        """
        pass
