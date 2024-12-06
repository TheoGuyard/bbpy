import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike

from bbpy.solver import Result, Status
from bbpy.instances.mip import CvxpyFunction, MIP, MIPBnB


class LinearFunction(CvxpyFunction):
    """Linear function of the form f(x) = Ax + b."""

    def __init__(self, A: ArrayLike, b: ArrayLike):
        self.A = A
        self.b = b

    def value(self, x):
        return self.A @ x + self.b

    def to_cvxpy_expr(self, x: cp.Variable):
        return self.A @ x + self.b


class QuadraticFunction(CvxpyFunction):
    """Quadratic function of the form f(x) = 1/2 x'Qx + c'x."""

    def __init__(self, Q: ArrayLike, c: ArrayLike):
        self.Q = Q
        self.c = c

    def value(self, x):
        return 0.5 * x @ (self.Q @ x) + self.c @ x

    def to_cvxpy_expr(self, x: cp.Variable):
        return 0.5 * cp.quad_form(x, self.Q) + self.c @ x


def generate_data_mip(n_var, n_con, n_int):
    """Generate data for a mixed-integer programming problem of the form:

    min 1/2 x'Qx + c'x
    s.t Ax <= b
        lb <= x <= ub
        x[i] in {0,1} for i in integers
    """

    Q = np.random.randn(n_var, n_var)
    Q = Q.T @ Q + np.eye(n_var)
    c = np.random.randn(n_var)
    A = np.random.randn(n_con, n_var)
    b = A @ np.ones(n_var)
    lb = np.full(n_var, -5.0)
    ub = np.full(n_var, +5.0)
    integers = np.random.choice(n_var, n_int, replace=False)
    lb[integers] = 0.0
    ub[integers] = 1.0

    f = QuadraticFunction(Q, c)
    g = LinearFunction(A, -b)

    return f, g, lb, ub, integers


def solve_mip_cvxpy(f, g, lb, ub, integers):

    x = cp.hstack(
        [
            cp.Variable(1, boolean=True) if i in integers else cp.Variable(1)
            for i in range(len(lb))
        ]
    )
    objective = cp.Minimize(f.to_cvxpy_expr(x))
    constraints = [g.to_cvxpy_expr(x) <= 0.0, lb <= x, x <= ub]
    problem = cp.Problem(objective, constraints)
    problem.solve(cp.SCIP)

    return Result(
        Status.OPTIMAL,
        x.value,
        problem.value,
        np.nan,
        -1,
        [],
    )


def test_mip():

    n_var = 50
    n_con = 25
    n_int = 10

    f, g, lb, ub, integers = generate_data_mip(n_var, n_con, n_int)

    problem = MIP(f, g, lb, ub, integers)
    solver = MIPBnB(verbose=True)
    result = solver.solve(problem)

    result_cvxpy = solve_mip_cvxpy(f, g, lb, ub, integers)

    error = np.abs(result.objective_value - result_cvxpy.objective_value)
    assert error < 1e-4, error
