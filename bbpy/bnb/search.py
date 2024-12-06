"""Tree search rules in the Branch-and-Bound algorithm."""

import random
from abc import ABC, abstractmethod

from .node import Node


class SearchingRule(ABC):
    """Base class for tree exploration rules."""

    @abstractmethod
    def add(self, queue: list, node: Node):
        """Add a new node to the queue."""
        pass

    @abstractmethod
    def get_next(self, queue: list):
        """Get the next node to explore in the queue."""
        pass


class BestFirstSearch(SearchingRule):
    """Best-First Search rule."""

    def add(self, queue: list, node: Node):
        queue.append(node)

    def get_next(self, queue: list):
        min_idx = min(range(len(queue)), key=lambda i: queue[i].lb)
        return queue.pop(min_idx)


class BreadthFirstSearch(SearchingRule):
    """Breadth-First Search rule."""

    def add(self, queue: list, node: Node):
        queue.append(node)

    def get_next(self, queue: list):
        return queue.pop(0)


class DepthFirstSearch(SearchingRule):
    """Depth-First Search rule."""

    def add(self, queue: list, node: Node):
        queue.append(node)

    def get_next(self, queue: list):
        return queue.pop()


class WorstFirstSearch(SearchingRule):
    """Worst-First Search rule."""

    def add(self, queue: list, node: Node):
        queue.append(node)

    def get_next(self, queue: list):
        min_idx = max(range(len(queue)), key=lambda i: queue[i].lb)
        return queue.pop(min_idx)


class RandomSearch(SearchingRule):
    """Random Search rule."""

    def add(self, queue: list, node: Node):
        queue.append(node)

    def get_next(self, queue: list):
        return queue.pop(random.randint(0, len(queue) - 1))
