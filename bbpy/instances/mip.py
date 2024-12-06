"""Mixed-integer programming problem and Branch-and-Bound solver."""

import cvxpy as cp
import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from typing import Type, Union
from numpy.typing import ArrayLike
from bbpy.problem import Problem
from bbpy.bnb import (
    BnB,
    LowerBoundingMethod,
    UpperBoundingMethod,
    BranchingRule,
    Node,
    Region,
    SearchingRule,
    BestFirstSearch,
)


class CvxpyFunction(ABC):
    """Base class for functions modeled using cvxpy."""

    @abstractmethod
    def value(self, x) -> float:
        """Value of the function at point x.

        Parameters
        ----------
        x : Any
            Point where the function is evaluated.
        """
        pass

    @abstractmethod
    def to_cvxpy_expr(self, x: cp.Variable) -> cp.Expression:
        """Convert the function to a cvxpy expression.

        Parameters
        ----------
        x : cp.Variable
            Variable used in the cvxpy expression.
        """
        pass


class MIP(Problem):
    """Mixed-integer programming problem of the form:

    min f(x)
    s.t g(x) <= 0
        lb <= x <= ub
        x[i] in {0,1} for i in integers

    where `f` is an extended real-valued function, `g` is an extended
    vector-valued function, `lb` and `ub` are the lower and upper bounds of the
    variables, and `integers` is the list of index of the variable that must be
    binary.

    Parameters
    ----------
    f : CvxpyFunction
        Objective function.
    g : CvxpyFunction
        Constraints.
    lb : Union[float, ArrayLike]
        Lower bounds of the variable.
    ub : Union[float, ArrayLike]
        Upper bounds of the variables.
    integers : list[int]
        Index of the variables that must be binary.
    """

    def __init__(
        self,
        f: CvxpyFunction,
        g: CvxpyFunction,
        lb: Union[float, ArrayLike],
        ub: Union[float, ArrayLike],
        integers: list[int],
    ):
        self.f = f
        self.g = g
        self.lb = lb
        self.ub = ub
        self.integers = integers

    def value(self, x: ArrayLike) -> float:
        if self.is_feasible(x):
            return self.f.value(x)
        else:
            return np.inf

    def is_feasible(self, x: ArrayLike, feas_tol: float = 1e-8) -> bool:
        cleq = np.all(np.all(self.g.value(x) <= feas_tol))
        cxlb = np.all(x >= self.lb - feas_tol)
        cxub = np.all(x <= self.ub + feas_tol)
        cbin = np.all(np.isin(x[self.integers], [0.0, 1.0]))
        return cleq and cxlb and cxub and cbin


class MIPRegion(Region):
    """Region of the feasible space for MIP problems where some binary
    entries of the variable are fixed to 0 or 1."""

    def __init__(self, integers_to_zero: list, integers_to_one: list):
        self.integers_to_zero = integers_to_zero
        self.integers_to_one = integers_to_one

    @classmethod
    def root_region(self, problem: MIP) -> Type["MIPRegion"]:
        return MIPRegion(integers_to_zero=[], integers_to_one=[])


class MIPBranchingRule(BranchingRule):
    """Branching rule for MIP problems based on maximal fractional entries in
    the lower-bounding solution."""

    def initialize(self, problem: MIP):
        self.integers = problem.integers

    def branch(self, node: Node) -> list[Node]:

        integers_to_zero = copy(node.region.integers_to_zero)
        integers_to_one = copy(node.region.integers_to_one)

        # Check if all binary variables are fixed
        if len(self.integers) == len(integers_to_zero) + len(integers_to_one):
            return []

        # Select maximal fractional variable
        i = sorted(
            set(self.integers) - set(integers_to_zero) - set(integers_to_one),
            key=lambda i: abs(node.x[i] - 0.5),
        )[0]

        # Child with new constraint x[i] = 0
        region0 = MIPRegion(
            integers_to_zero=integers_to_zero + [i],
            integers_to_one=copy(integers_to_one),
        )
        x0 = np.copy(node.x)
        x0[i] = 0.0
        child0 = Node(region=region0, level=node.level + 1, x=x0, lb=node.lb)

        # Child with new constraint x[i] = 1
        region1 = MIPRegion(
            integers_to_zero=copy(integers_to_zero),
            integers_to_one=integers_to_one + [i],
        )
        x1 = np.copy(node.x)
        x1[i] = 1.0
        child1 = Node(region=region1, level=node.level + 1, x=x1, lb=node.lb)

        return [child0, child1]


class MIPLowerBoundingMethod(LowerBoundingMethod):
    """Lower bounding method for MIP problems with a continuous relaxation
    of the binary variables. The lower bounding problem is solved using the
    SCIP solver via the cvxpy interface."""

    def initialize(self, problem: MIP):
        self.problem = problem
        self.x = cp.Variable(len(problem.lb))
        self.objective = cp.Minimize(problem.f.to_cvxpy_expr(self.x))
        self.constraints = [
            problem.g.to_cvxpy_expr(self.x) <= 0.0,
            self.x >= problem.lb,
            self.x <= problem.ub,
        ]

    def bound(self, node: Node):

        # Node constraints
        node_constraints = []
        for i in self.problem.integers:
            if i in node.region.integers_to_zero:
                node_constraints.append(self.x[i] == 0.0)
            elif i in node.region.integers_to_one:
                node_constraints.append(self.x[i] == 1.0)
            else:
                node_constraints.append(self.x[i] >= 0.0)
                node_constraints.append(self.x[i] <= 1.0)

        # Instantiate and solve the lower bounding problem
        problem = cp.Problem(
            self.objective,
            self.constraints + node_constraints,
        )
        result = problem.solve(cp.SCIP)

        # Update node attributes
        if problem.status in {"optimal"}:
            node.x = self.x.value
            node.lb = result
        else:
            node.x = self.x.value
            node.lb = -np.inf


class MIPUpperBoundingMethod(UpperBoundingMethod):
    """Upper bounding method for MIP problems where an incumbent is generated
    forcing the binary variables to the closest integer based on the lower
    bounding solution."""

    def initialize(self, problem: MIP):
        self.problem = problem
        self.x = cp.Variable(len(problem.lb))
        self.objective = cp.Minimize(problem.f.to_cvxpy_expr(self.x))
        self.constraints = [
            problem.g.to_cvxpy_expr(self.x) <= 0.0,
            self.x >= problem.lb,
            self.x <= problem.ub,
        ]

    def bound(self, node: Node):

        # Set binary entries to closest integer in the lower bounding solution
        integer_constraints = []
        for i in self.problem.integers:
            xi = np.round(np.clip(node.x[i], 0.0, 1.0))
            integer_constraints.append(self.x[i] == xi)

        # Instantiate and solve the lower bounding problem
        problem = cp.Problem(
            self.objective,
            self.constraints + integer_constraints,
        )
        problem.solve(cp.SCIP)

        if problem.status in {"optimal"}:
            x = self.x.value
        else:
            x = copy(node.x)

        x[self.problem.integers] = np.clip(x[self.problem.integers], 0.0, 1.0)
        x[self.problem.integers] = np.round(x[self.problem.integers])

        return x


class MIPBnB(BnB):
    """Branch-and-bound implementation for MIP problems."""

    def default_region(self) -> Region:
        return MIPRegion(integers_to_zero=[], integers_to_one=[])

    def default_lower_bounding_method(self) -> LowerBoundingMethod:
        return MIPLowerBoundingMethod()

    def default_upper_bounding_method(self) -> UpperBoundingMethod:
        return MIPUpperBoundingMethod()

    def default_branching_rule(self) -> BranchingRule:
        return MIPBranchingRule()

    def default_searching_rule(self) -> SearchingRule:
        return BestFirstSearch()
