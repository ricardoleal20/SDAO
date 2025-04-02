"""
Include the Two Step SQP algorithm as one of the
State of Art (SoA) algorithms to compare with the
SDAO algorithm.

Link: https://arxiv.org/abs/2408.16656
"""

import numpy as np
from scipy.optimize import minimize


class TwoStepSQP:
    """Two Stepsize SQP Method for Stochastic Optimization."""

    def __init__(self, obj_func, constraints, grad_func=None, hessian_func=None):
        """Initialize the TwoStepSQP optimization class.

        Parameters:
        - obj_func: Callable, objective function to minimize.
        - constraints: List of dicts, nonlinear constraints in scipy format.
        - grad_func: Callable, gradient of the objective function (optional).
        - hessian_func: Callable, Hessian of the objective function (optional).
        """
        self.obj_func = obj_func
        self.constraints = constraints
        self.grad_func = grad_func
        self.hessian_func = hessian_func

    def optimize(self, x0, max_iter=100, tol=1e-6):
        """Perform the two-step SQP optimization.

        Parameters:
        - x0: ndarray, initial guess for the solution.
        - max_iter: int, maximum number of iterations.
        - tol: float, tolerance for convergence.

        Returns:
        - result: scipy.optimize.OptimizeResult
        """
        x = x0

        for iteration in range(max_iter):
            # Step 1: Quadratic subproblem to determine step direction
            def subproblem_obj(dx):
                return self.obj_func(x + dx)

            def subproblem_grad(dx):
                return self.grad_func(x + dx) if self.grad_func else None

            quadratic_constraints = [
                {
                    "type": con["type"],
                    "fun": lambda dx, c=con: c["fun"](x + dx),
                    "jac": lambda dx, c=con: c["jac"](x + dx) if "jac" in c else None,
                }
                for con in self.constraints
            ]

            result = minimize(
                subproblem_obj,
                np.zeros_like(x),
                jac=subproblem_grad,
                constraints=quadratic_constraints,
                method="SLSQP",
                options={"ftol": tol, "disp": False},
            )

            if not result.success:
                print(f"Subproblem optimization failed at iteration {iteration}")
                break

            dx = result.x

            # Step 2: Line search to determine step length
            alpha = self.line_search(x, dx)
            x_new = x + alpha * dx

            # Check for convergence
            if np.linalg.norm(x_new - x) < tol:
                print(f"Converged at iteration {iteration}")
                break

            x = x_new

        return x

    def line_search(self, x, dx, alpha_init=1.0, rho=0.5, c=1e-4):
        """Perform a backtracking line search.

        Parameters:
        - x: ndarray, current position.
        - dx: ndarray, direction vector.
        - alpha_init: float, initial step size.
        - rho: float, contraction factor for step size.
        - c: float, sufficient decrease constant.

        Returns:
        - alpha: float, step size.
        """
        alpha = alpha_init

        while self.obj_func(x + alpha * dx) > self.obj_func(x) + c * alpha * np.dot(
            dx, dx
        ):
            alpha *= rho

        return alpha
