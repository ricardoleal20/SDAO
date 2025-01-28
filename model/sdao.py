"""
Implement the SDAO model in Python.
"""
from typing import TypedDict, Callable, Sequence, Literal
from dataclasses import dataclass
import numpy as np
# Local imports
from model.soa.template import Algorithm


class SDAOParams(TypedDict):
    """Parameters for the SDAO model.
    
    - learning_rate (alpha): Learning rate of the algorithm. Controls
        the step size of the particles in the search space.
    - memory_coeff (gamma): Memory coefficient of the algorithm. This
        coefficient is used to adjust the weight of the best position
        in the search of solutions.
    - decay_rate (beta): Decay rate of the algorithm. This is used during
        the update of the coefficient of diffusion.
    - diffusion_coeff (D): The initial diffusion coefficient of the algorithm.
    """
    learning_rate: float
    memory_coeff: float
    diffusion_coeff: float
    decay_rate: float


@dataclass
class Particle:
    """Particle used in the SDAO model to represent a possible solution."""
    position: np.ndarray
    value: float  # The value of the objective function for this particle
    best_value: float  # The best value found in the particle
    best_position: np.ndarray  # The best position found by this particle


class SDAO(Algorithm):
    """Stochastic Diffusion Adaptive Optimization (SDAO) model.
    
    This model involves the following parameters to adjust:
    - num_particles: Number of particles to use.
    - dimension: Dimension of the search space.
    - learning_rate: Learning rate of the algorithm.
    - memory_coeff: Memory coefficient of the algorithm.
    - diffusion_coeff: Diffusion coefficient of the algorithm.
    - decay_rate: Decay rate of the algorithm.
    - max_iterations: Maximum number of iterations.
    - objective_fn: Objective function to optimize.
    - threshold: Threshold to consider the algorithm has converged. Default to 1e-6.
    - num_threads: Number of threads to use. Default to 1.
    - verbose: Whether to print information during the run. Default to false.
    """
    _n_part: int
    _n_iter: int
    _params: SDAOParams
    _v: bool
    _version: Literal[0, 1, 2]
    # Slots of the class
    __slots__ = [
        "_params", "_n_part", "_v", "_n_iter", "_version"
    ]

    def __init__(  # pylint: disable=R0913
        self,
        num_particles: int = 30,
        num_iterations: int = 100,
        version: Literal[0, 1, 2] = 0,
        *,
        params: SDAOParams,
        verbose: bool = False,
    ) -> None:
        self._n_part = num_particles
        self._n_iter = num_iterations
        # Initialize the parameters...
        self._params = params
        self._v = verbose
        self._version = version

    # ====================================== #
    #              Public methods            #
    # ====================================== #

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using the SDAO algorithm."""
        # Initialize the particles...
        particles = _init_particles(self._n_part, dimension, bounds)
        # Init the first value of each the particles using the objective function
        for particle in particles:
            obj_value = objective_fn(particle.position)
            particle.value = obj_value
            particle.best_value = obj_value
            particle.best_position = particle.position
        # With this, initialize the diff_coeff as the parameter given in the SDAO params
        diff_coeff = self._params["diffusion_coeff"]
        # Run it using the maximum number of iterations
        improvement = False
        for k in range(self._n_iter):
            # Update each particle
            for particle in particles:
                # Update the particle
                particle, improvement = self.__update_particle(
                    particle, objective_fn, diff_coeff, bounds)
                # Check the optimallity of the particle
                # ! NOTE: To be implemented
            # Update the diffusion coefficient
            if improvement:
                diff_coeff *= np.exp(-self._params["decay_rate"] * k)
            else:
                diff_coeff *= np.exp(-self._params["memory_coeff"])
        # Return the best particle found
        best_particle = min(particles, key=lambda x: x.best_value)
        return best_particle.best_value, best_particle.best_position

    # ====================================== #
    #             Private methods            #
    # ====================================== #
    def __update_particle(
        self,
        particle: Particle,
        objective_fn: Callable[[np.ndarray], float | int],
        diff_coeff: float,
        bounds: Sequence[tuple[float, float]] | tuple[float, float]
    ) -> tuple[Particle, bool]:
        """Get the new particle position and value."""
        # With this, update the position
        # Using...
        # P1: Disturbance term
        disturbance_term = self.__get_disturbance_term(particle)
        # P2: Memory term
        memory_term = self._params["memory_coeff"] * \
            (particle.best_position - particle.position)
        # P3: Noisy diffusion term
        # Calculate the stoch term using a Heavy-tailed distribution
        stoch_term = np.sqrt(2 * diff_coeff) * \
            np.random.laplace(0, 1, particle.position.shape)

        new_position = particle.position \
            + disturbance_term + memory_term + stoch_term
        # ANY POSITION OUTSIDE THE BOUNDS WILL BE MOVED TO THE BOUNDARY
        if isinstance(bounds, tuple):
            new_position = np.clip(new_position, *bounds)  # type: ignore
        else:
            for i, d in enumerate(bounds):
                new_position[i] = np.clip(new_position[i], *d)  # type: ignore

        new_position = np.clip(new_position, -1, 1)
        # Update the particle position
        particle.position = new_position
        particle.value = objective_fn(new_position)
        # Update the new best position if the value is better
        improvement: bool = False
        if particle.value < particle.best_value:
            particle.best_value = particle.value
            particle.best_position = new_position
            improvement = True
        return particle, improvement

    def __get_disturbance_term(
        self,
        particle: Particle,
    ) -> np.ndarray | float:
        """Get the disturbance term for the particle."""
        # Depending on the SDAO version, we'll use different methods...
        match self._version:
            case 0:
                # This version still use the gradient of the function for this
                # particle
                grad = np.gradient(particle.value)
                if grad:
                    return -self._params["learning_rate"] * grad
                return np.zeros(particle.position.shape)
            case 1:
                # Use the method of:
                #  * alpha * sign(f(x) - f(x_best))
                #  * sign(f(x) - f(x_best)) = 1 if f(x) < f(x_best) else -1
                sign = 1 if particle.value < particle.best_value else -1
                return self._params["learning_rate"] * sign
            case 2:
                # In this case, we use an adaptive noise term
                noise = np.random.normal(0, 1, size=2)
                sign = 1 if noise[0] < noise[1] else -1
                return self._params["learning_rate"] * sign
            case _:
                raise ValueError(
                    f"Invalid version for the SDAO model. Version: {self._version}"
                )

# ====================================== #
#              Helper methods            #
# ====================================== #


def _init_particles(
    num_particles: int,
    dimension: int,
    bounds: Sequence[tuple[float, float]] | tuple[float, float]
) -> list[Particle]:
    """Initialize the particles for the SDAO algorithm."""
    return [Particle(
        position=np.random.uniform(*bounds, size=dimension),
        value=np.inf,
        best_value=np.inf,
        best_position=np.zeros(dimension)
    ) for _ in range(num_particles)]
