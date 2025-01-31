"""
Implement the SDAO model in Python.
"""
from typing import TypedDict, Callable, Sequence, Literal, Optional
from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree
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
    _best_part: Particle
    # Slots of the class
    __slots__ = [
        "_params", "_n_part", "_v", "_n_iter", "_version",
        "_best_part",
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
        #! DELETE
        self._best_part = None  # type: ignore

    # ====================================== #
    #              Public methods            #
    # ====================================== #

    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
        # Índices de variables discretas
        discrete_vars: Optional[list[int]] = None
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using the SDAO algorithm."""
        # Initialize the particles...
        particles = _init_particles(self._n_part, dimension, bounds)

        # ! NOTE: THIS IS TEMPORAL FOR THE TESTS
        delta = 0.1
        gamma_low = 0.3
        # Init the first value of each the particles using the objective function
        for particle in particles:
            obj_value = objective_fn(particle.position)
            particle.value = obj_value
            particle.best_value = obj_value
            particle.best_position = particle.position.copy()
        # Get the best particle
        self._best_part = min(particles, key=lambda x: x.best_value)
        # Construir KDTree para la búsqueda eficiente de vecinos
        kdtree = KDTree([p.position for p in particles])
        # With this, initialize the diff_coeff as the parameter given in the SDAO params
        diff_coeff = self._params["diffusion_coeff"]
        # Run it using the maximum number of iterations
        for k in range(self._n_iter):
            # Calculate the diversity of the swarm
            swarm_diversity = _calc_swarm_diversity(particles)
            memory_k = self._params["memory_coeff"] if swarm_diversity > delta else gamma_low
            # Update each particle
            improvement_flag = False
            for particle in particles:
                # Update the particle
                particle, improvement = self.__update_particle(
                    particle, objective_fn,
                    diff_coeff, memory_k, bounds,
                    kdtree, discrete_vars
                )
                # Update the improvement flag
                improvement_flag = improvement_flag or improvement
            # Update the diffusion coefficient
            if improvement_flag:
                diff_coeff *= np.exp(-self._params["decay_rate"] * k)
            else:
                diff_coeff *= np.exp(-self._params["memory_coeff"])
            # Actualizar el KDTree con las nuevas posiciones
            kdtree = KDTree([p.position for p in particles])

            # Get the best particle
            current_best_part = min(particles, key=lambda x: x.best_value)
            # Evalaute if this is better than the current best particle
            if current_best_part.best_value < self._best_part.best_value:
                self._best_part = current_best_part

            # Optional: Print progress ONLY if verbose is enabled
            if self._v and (k % 10 == 0 or k == self._n_iter - 1):
                print(
                    f"Iteración {k}: Mejor Valor = {self._best_part.best_value}")

        # Return the best particle found
        return self._best_part.best_value, self._best_part.best_position

    # ====================================== #
    #             Private methods            #
    # ====================================== #
    def __update_particle(  # pylint: disable=R0913, R0914
        self,
        particle: Particle,
        objective_fn: Callable[[np.ndarray], float | int],
        diff_coeff: float,
        memory_coeff: float,
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        kdtree: KDTree,
        discrete_vars: Optional[list[int]]
    ) -> tuple[Particle, bool]:
        """Get the new particle position and value."""
        # P1: Disturbance term
        disturbance_term = self.__get_disturbance_term(particle)
        disturbance_term = 0
        # P2: Memory term
        memory_term = memory_coeff * \
            (particle.best_position - particle.position)
        # P3: Noisy diffusion term
        # Calculate the stoch term using a Heavy-tailed distribution
        stoch_term = np.sqrt(2 * diff_coeff) * \
            np.random.laplace(0, 1, particle.position.shape)
        # # Global memory term
        global_attraction = 0.2 * \
            (self._best_part.best_position - particle.position)
        # Get the diffusion term INSPIRED on the 2nd Fick's Law
        density_term = self.__get_diffusion_term(particle, kdtree)

        new_position = particle.position \
            + disturbance_term + memory_term + stoch_term \
            + density_term + global_attraction
        # ANY POSITION OUTSIDE THE BOUNDS WILL BE MOVED TO THE BOUNDARY
        if isinstance(bounds, tuple):
            new_position = np.clip(new_position, *bounds)  # type: ignore
        else:
            for i, d in enumerate(bounds):
                new_position[i] = np.clip(new_position[i], *d)  # type: ignore

        # Si hay variables discretas, aplicarlas
        if discrete_vars:
            for idx in discrete_vars:
                # Por ejemplo, redondear a entero más cercano
                new_position[idx] = np.round(new_position[idx])
            # Opcional: Reparar posiciones que exceden los límites después de la discretización
            if isinstance(bounds, tuple):
                new_position = np.clip(new_position, *bounds)  # type: ignore
            else:
                for i, d in enumerate(bounds):
                    new_position[i] = np.clip(
                        new_position[i], *d)  # type: ignore

        # Actualizar la posición de la partícula
        particle.position = new_position
        particle.value = objective_fn(new_position)
        # Update the best position if the value is better
        improvement: bool = False
        if particle.value < particle.best_value:
            particle.best_value = particle.value
            particle.best_position = new_position.copy()
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
                if grad.any():
                    return -self._params["learning_rate"] * grad
                return np.zeros(particle.position.shape)
            case 1:
                # Use the method of:
                #  * alpha * sign(f(x) - f(x_best))
                #  * sign(f(x) - f(x_best)) = 1 if f(x) < f(x_best) else -1
                sign = 1 if particle.value < particle.best_value else -1
                return self._params["learning_rate"] * sign
            case 2:
                # Use adaptive noise term
                noise = np.random.normal(0, 1, size=particle.position.shape)
                sign = 1 if noise[0] < noise[1] else -1
                return self._params["learning_rate"] * sign
            case _:
                raise ValueError(
                    f"Invalid version for the SDAO model. Version: {self._version}"
                )

    def __get_diffusion_term(
        self,
        particle: Particle,
        kdtree: KDTree
    ) -> np.ndarray:
        """Calculate the diffusion term inspired by the 2nd Fick's Law.
        This term represents the tendency of particles to move from
        high-density areas to low-density areas.
        """
        # Get the diffusion parameters, such as the k-radius for the tree
        radius = 1.0  # * Note: This could be an adaptive parameter...
        neighbors_idx = kdtree.query_ball_point(particle.position, r=radius)
        neighbors_idx = [idx for idx in neighbors_idx if not np.array_equal(
            particle.position, kdtree.data[idx])]

        if not neighbors_idx:
            return np.zeros_like(particle.position)
        # Calculate the center of mass of the neighbors
        neighbors_pos = kdtree.data[neighbors_idx]
        mean_neighbor_pos = np.mean(neighbors_pos, axis=0)

        # Calculate the density gradient (direction towards low density)
        density_gradient = particle.position - mean_neighbor_pos
        norm = np.linalg.norm(density_gradient)
        if norm == 0:
            return np.zeros_like(particle.position)
        density_gradient /= norm  # Normalize the gradient

        # Ensure the gradient is in the right direction by scaling it
        return density_gradient * self._params["diffusion_coeff"]

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


def _calc_swarm_diversity(particles: list[Particle]) -> float:
    """Compute a simple measure of swarm diversity:
       * The average of the dimension-wise standard deviations.
    """
    positions = np.array([p.position for p in particles]
                         )  # shape: (num_particles, dimension)
    # standard deviation per dimension
    std_dev = np.std(positions, axis=0)  # shape: (dimension,)
    # average across all dimensions
    return float(np.mean(std_dev))
