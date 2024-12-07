"""Branch-and-Bound solver."""

import sys
import time
from abc import abstractmethod
from copy import copy
from typing import Union
from bbpy.problem import Problem
from bbpy.solver import Result, Solver, Status

from .bound import LowerBoundingMethod, UpperBoundingMethod
from .branch import BranchingRule
from .node import Node, NodeStatus, Region
from .search import SearchingRule


class BnB(Solver):
    """Branch-and-Bound solver.

    Parameters
    ----------
    verbose : bool
        Toggle displays. Default is False.
    keeptrace : bool
        Toggle trace keeping. Default is False.
    time_limit : float
        The time limit in seconds. Default is infinity.
    node_limit : int
        The node limit. Default is infinity.
    queue_limit : int
        The maximum queue size. Default is infinity.
    feas_tol : float
        The feasibility tolerance for constraints. Default is 1e-8.
    abs_gap_tol : float
        The absolute gap tolerance on the optimal value. Default is 1e-8.
    rel_gap_tol : float
        The relative gap tolerance on the optimal value. Default is 1e-8.
    lb_method : Union[LowerBoundingMethod, None]
        The lower bounding method. Default is None and will be set
        according to the `default_lower_bounding_method` method.
    ub_method : Union[CandidateSolver, None]
        The upper bounding method. Default is None and will be set
        according to the `default_upper_bounding_method` method.
    branching_rule : Union[BranchingRule, None]
        Tree branching rule. Default is None and will be set according to the
        `default_branching_rule` method.
    searching_rule : Union[SearchingRule, None]
        Tree searching rule. Default is None and will be set according to the
        `default_searching_rule` method.
    """

    def __init__(
        self,
        verbose: bool = False,
        keeptrace: bool = False,
        time_limit: float = float("inf"),
        node_limit: int = sys.maxsize,
        queue_limit: int = sys.maxsize,
        feas_tol: float = 1e-8,
        abs_gap_tol: float = 1e-8,
        rel_gap_tol: float = 1e-8,
        region: Union[Region, None] = None,
        lb_method: Union[LowerBoundingMethod, None] = None,
        ub_method: Union[UpperBoundingMethod, None] = None,
        branching_rule: Union[BranchingRule, None] = None,
        searching_rule: Union[SearchingRule, None] = None,
    ) -> None:
        self.verbose = verbose
        self.keeptrace = keeptrace
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.queue_limit = queue_limit
        self.feas_tol = feas_tol
        self.abs_gap_tol = abs_gap_tol
        self.rel_gap_tol = rel_gap_tol

        if region is None:
            self.region = self.default_region()
        else:
            self.region = region

        if lb_method is None:
            self.lb_method = self.default_lower_bounding_method()
        else:
            self.lb_method = lb_method

        if ub_method is None:
            self.ub_method = self.default_upper_bounding_method()
        else:
            self.ub_method = ub_method

        if branching_rule is None:
            self.branching_rule = self.default_branching_rule()
        else:
            self.branching_rule = branching_rule

        if searching_rule is None:
            self.searching_rule = self.default_searching_rule()
        else:
            self.searching_rule = searching_rule

    @abstractmethod
    def default_region(self) -> Region:
        """Default region design for the problem."""
        pass

    @abstractmethod
    def default_lower_bounding_method(self) -> LowerBoundingMethod:
        """Default lower bounding method."""
        pass

    @abstractmethod
    def default_upper_bounding_method(self) -> UpperBoundingMethod:
        """Default upper bounding method."""
        pass

    @abstractmethod
    def default_branching_rule(self) -> BranchingRule:
        """Default branching rule."""
        pass

    @abstractmethod
    def default_searching_rule(self) -> SearchingRule:
        """Default searching rule."""
        pass

    def _display_header(self) -> None:
        print("-" * 80)
        print(
            "{:>6} | {:>6} | {:>6} | {:>9} | {:>8} | {:>8} | {:>8} | {:>8}".format(  # noqa: E501
                "timer",
                "tree",
                "queue",
                "node",
                "ub",
                "lb",
                "agap",
                "rgap",
            )
        )
        print("-" * 80)

    def _display_inner(self, node: Node) -> None:
        print(
            "{:>6.2f} | {:>6} | {:>6} | {:>9} | {:>8.4f} | {:>8.4f} | {:>8.2e} | {:>8.2e}".format(  # noqa: E501
                self.timer,
                self.node_count,
                len(self.queue),
                node.status.value,
                self.best_ub,
                self.best_lb,
                self.abs_gap,
                self.rel_gap,
            )
        )

    def _display_footer(self) -> None:
        print("-" * 80)

    def _abs_gap(self, ub: float, lb: float) -> float:
        return ub - lb

    def _rel_gap(self, ub: float, lb: float) -> float:
        return (ub - lb) / max(min(abs(ub), abs(lb)), 1e-16)

    def solve(self, problem: Problem) -> Result:

        # Initialize BnB attributes
        self.start_time = time.time()
        self.node_count = 0
        self.best_x = None
        self.best_ub = float("inf")
        self.best_lb = -float("inf")
        self.queue = []
        self.trace = []
        self.status = Status.RUNNING

        # Initialize BnB components
        self.lb_method.initialize(problem)
        self.ub_method.initialize(problem)
        self.branching_rule.initialize(problem)

        # Initialize BnB tree
        root = Node(region=self.region.root_region(problem), level=0)
        self.searching_rule.add(self.queue, root)

        if self.verbose:
            self._display_header()

        # Main loop
        while self.status == Status.RUNNING:

            # Next node selection
            node = self.searching_rule.get_next(self.queue)

            # Lower bound evaluation
            self.lb_method.bound(node)

            # Pruning test
            if node.lb > self.best_ub:
                node.status = NodeStatus.PRUNED
            else:
                node.status = NodeStatus.PROCESSED

                # Upper bound evaluation
                if problem.is_feasible(node.x, feas_tol=self.feas_tol):
                    x = copy(node.x)
                    ub = node.lb
                else:
                    x = self.ub_method.bound(node)
                    ub = problem.value(x)

                # Upper bound update
                if ub < self.best_ub:
                    node.status = NodeStatus.FEASIBLE
                    self.best_x = x
                    self.best_ub = ub
                    self.queue = [
                        qnode
                        for qnode in self.queue
                        if qnode.lb <= self.best_ub
                    ]

                # Branching operation
                if (
                    self._abs_gap(self.best_ub, node.lb) > self.abs_gap_tol
                    and self._rel_gap(self.best_ub, node.lb) > self.rel_gap_tol
                ):
                    child_nodes = self.branching_rule.branch(node)
                    self.queue.extend(child_nodes)

            # Lower bound update
            if len(self.queue):
                self.best_lb = min(qnode.lb for qnode in self.queue)
            else:
                self.best_lb = self.best_ub

            # Working values update
            self.node_count += 1
            self.abs_gap = self._abs_gap(self.best_ub, self.best_lb)
            self.rel_gap = self._rel_gap(self.best_ub, self.best_lb)
            self.timer = time.time() - self.start_time

            # Trace update
            if self.keeptrace:
                self.trace.append(
                    {
                        "node_count": self.node_count,
                        "timer": self.timer,
                        "queue_size": len(self.queue),
                        "best_ub": self.best_ub,
                        "best_lb": self.best_lb,
                        "node": node.trace,
                    }
                )

            # Displays
            if self.verbose:
                self._display_inner(node)

            # Status update
            if self.timer >= self.time_limit:
                self.status = Status.TIME_LIMIT
            if self.node_count >= self.node_limit:
                self.status = Status.ITER_LIMIT
            if len(self.queue) >= self.queue_limit:
                self.status = Status.MEMORY_LIMIT
            if self.abs_gap <= self.abs_gap_tol:
                self.status = Status.RELATIVE_OPTIMAL
            if self.rel_gap <= self.rel_gap_tol:
                self.status = Status.RELATIVE_OPTIMAL
            if not len(self.queue):
                self.status = Status.OPTIMAL

        if self.verbose:
            self._display_footer()

        return Result(
            self.status,
            self.best_x,
            self.best_ub,
            self.timer,
            self.node_count,
            self.trace,
        )
