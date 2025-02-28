"""
Implement the SDAO model in Python.
"""
from typing import TypedDict, Callable, Sequence, Literal
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
    stagnation_counter: int  # Counter for consecutive iterations without improvement



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
        "_params", "_n_part", "_v", "_n_iter",
        "_version", "_best_part",
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
        self._version = version  # deprecated!
        # Add the best particule as None
        self._best_part = None  # type: ignore

    # ====================================== #
    #              Public methods            #
    # ====================================== #

    def optimize(  # pylint: disable=R0914
        self,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        dimension: int,
    ) -> tuple[float, np.ndarray]:
        """Optimize the objective function using the SDAO algorithm."""
        # Save original bounds for later contraction
        original_bounds = bounds if isinstance(bounds, tuple) else list(bounds)

        # Initialize the particles...
        particles = _init_particles(self._n_part, dimension, bounds)

        # Init the first value of each the particles using the objective function
        for particle in particles:
            obj_value = objective_fn(particle.position)
            particle.value = obj_value
            particle.best_value = obj_value
            particle.best_position = particle.position.copy()
        # Apply OBL during initialization
        particles = [self.__apply_obl(
            particle, objective_fn, bounds)
            for particle in particles
        ]

        # Get the best particle
        self._best_part = min(particles, key=lambda x: x.best_value)
        # Build the KDTree for efficient neighbor search
        kdtree = KDTree([p.position for p in particles])
        # With this, initialize the diff_coeff as the parameter given in the SDAO params
        diff_coeff = self._params["diffusion_coeff"]
        # Run it using the maximum number of iterations
        for k in range(self._n_iter):
            # Update each particle
            improvement_flag = False
            for particle in particles:
                # Update the particle
                particle, improvement = self.__update_particle(
                    particle, objective_fn,
                    diff_coeff, bounds,
                    kdtree
                )
                # Based on the improvement, update the stagnation counter
                # of the particle
                if improvement:
                    # If the particle improved, reset the stagnation counter
                    # to zero, this to reduce their probability of applying OBL
                    # in the next iterations.
                    particle.stagnation_counter = 0
                else:
                    # Add one to the stagnation counter
                    particle.stagnation_counter += 1
                    # * Apply OBL based on the stagnation counter.
                    # * The probability of applying OBL is defined as: P = 1 - exp(-λ * SC)
                    # * where SC is the stagnation counter of this specific particle.
                    prob_obl = 1 - \
                        np.exp(-self._params["decay_rate"]
                               * particle.stagnation_counter)
                    # * Generate a random number in [0, 1] to decide if OBL is applied.
                    if np.random.rand() < prob_obl:
                        particle = self.__apply_obl(
                            particle, objective_fn, bounds)

                # Update the particle improvement flag if needed
                improvement_flag = improvement_flag or improvement

            # Get the swarm diversity for the particles and their global density
            diversity = _calc_swarm_diversity(particles)
            global_density = _calc_global_density(particles)

            delta_min = self._params["diffusion_coeff"] * 0.1
            delta_max = self._params["diffusion_coeff"]
            # Calculate the weight for the transition between methods
            if diversity <= delta_min:
                # If the diversity is below the minimum, the weight is 1.0
                weight = 1.0
            elif diversity >= delta_max:
                # Otherwise, if the diversity is higher than the maximum,
                # the weight is 0.0
                weight = 0.0
            else:
                # And, if the diversity is between the minimum and maximum,
                # we calculate the weight using the formula:
                # w = (delta_max - diversity) / (delta_max - delta_min)
                weight = (delta_max - diversity) / (delta_max - delta_min)

            # * Update the diffusion coefficient for the next iteration...
            # Method 1: Time-based decay
            diff_time_based = self._params["diffusion_coeff"] * \
                np.exp(-self._params["decay_rate"] * k)
            # Method 2: Improved decay with density
            diff_density_based = diff_time_based * (1 + 0.5 * global_density)
            # Dynamic selection of the diffusion coefficient
            diff_coeff = diff_time_based + weight * \
                (diff_density_based - diff_time_based)
            # # Every fixed number of iterations, contract the domain
            if (k + 1) % 20 == 0:  # Do it every 20 iterations ! Maybe a parameter?
                bounds = self.__contract_bounds(
                    original_bounds, self._best_part.best_position)

            # Update the KD Tree for the new positions of the particles
            kdtree = KDTree([p.position for p in particles])
            # Get the best particle
            current_best_part = min(particles, key=lambda x: x.best_value)
            # Evalaute if this is better than the current best particle
            if current_best_part.best_value < self._best_part.best_value:
                self._best_part = current_best_part

            # Optional: Print progress ONLY if verbose is enabled
            if self._v and (k % 10 == 0 or k == self._n_iter - 1):
                print(
                    f"Iteration [{k}] | Best Value = {self._best_part.best_value}")

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
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
        kdtree: KDTree,
    ) -> tuple[Particle, bool]:
        """Get the new particle position and value.
        
        For this, we're going to use the following equation:

        x_i^{k+1} = (
            + x_i^k : Current position of the particle.
            + D(x_i^k) : Diffusion term (inspired by 2nd Fick's Law).
            + delta * (x:{global best} - x_i^k) : Global attraction term
            + gamma * (x_{best} - x_i^k) : Memory term for the particle
            + sqrt(2 * D) * Laplace(0, 1) : Noisy diffusion term
        )
        """
        # P1: Diffusion term, directly inspired by 2nd Fick's Law.
        # Along with this, get a global attraction term. This is going to
        # replace the Gradient calculation in the original SDAO model.
        # * Density term
        density_term = self.__get_diffusion_term(particle, kdtree)
        # * Global memory term
        global_attraction = self._params["learning_rate"] * \
            (self._best_part.best_position - particle.position)
        # P2: Memory term
        memory_term = self._params["memory_coeff"] * \
            (particle.best_position - particle.position)
        # P3: Noisy diffusion term
        # Calculate the stoch term using a Heavy-tailed distribution
        stoch_term = np.sqrt(2 * diff_coeff) * \
            np.random.uniform(0, 1, particle.position.shape)
        # * Update the position of the particle
        new_position = particle.position \
            + density_term + global_attraction \
            + memory_term + stoch_term

        # ANY POSITION OUTSIDE THE BOUNDS WILL BE MOVED TO THE BOUNDARY
        if isinstance(bounds, tuple):
            new_position = np.clip(new_position, *bounds)  # type: ignore
        else:
            for i, d in enumerate(bounds):
                new_position[i] = np.clip(new_position[i], *d)  # type: ignore

        # Update the particle position and value
        particle.position = new_position
        particle.value = objective_fn(new_position)
        # Update the best position if the value is better
        improvement: bool = False
        if particle.value < particle.best_value:
            particle.best_value = particle.value
            particle.best_position = new_position.copy()
            improvement = True
        return particle, improvement

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

    def __apply_obl(
        self,
        particle: Particle,
        objective_fn: Callable[[np.ndarray], float | int],
        bounds: Sequence[tuple[float, float]] | tuple[float, float],
    ) -> Particle:
        """Apply Opposition-Based Learning (OBL) for a single particle.
        
        The OBL is a technique that generates the opposite position of a
        particle and evaluates it. If the opposite position is better than
        the current position, the particle is updated.
        """
        # Generate the opposite position
        opposite_position = self.__generate_opposite_position(
            particle.position, bounds)
        # Evaluate the opposite position
        opposite_value = objective_fn(opposite_position)
        # Compare and keep the better position
        if opposite_value < particle.value:
            particle.position = opposite_position
            particle.value = opposite_value
            # Update the best position if necessary
            if opposite_value < particle.best_value:
                particle.best_position = opposite_position.copy()
                particle.best_value = opposite_value
        return particle

    def __generate_opposite_position(
        self,
        position: np.ndarray,
        bounds: Sequence[tuple[float, float]] | tuple[float, float]
    ) -> np.ndarray:
        """Generate the opposite position for a given particle position."""
        opposite_position = np.empty_like(position)
        if isinstance(bounds, tuple):
            lower, upper = bounds
            opposite_position = lower + upper - position  # type: ignore
        else:
            for i, (lower, upper) in enumerate(bounds):
                opposite_position[i] = lower + upper - position[i]
        return opposite_position

    def __contract_bounds(
        self,
        original_bounds: Sequence[tuple[float, float]] | tuple[float, float],
        best_position: np.ndarray
    ) -> Sequence[tuple[float, float]]:
        """Contract the search bounds based on the best global solution.
        
        For each dimension j:
          new_lower_j = best_position_j - delta * (best_position_j - original_lower_j)
          new_upper_j = best_position_j + delta * (original_upper_j - best_position_j)
        """
        # Create the array for the new bounds...
        new_bounds = []
        # The bounds are the same for all dimensions, so we modify them as a list
        if isinstance(original_bounds, tuple):
            bounds = [original_bounds]  # type: ignore
            was_tuple: bool = True
        else:
            bounds: list[tuple[float, float]] = original_bounds  # type: ignore
            was_tuple = False
        # We iterate over the bounds to update them. We are not going further the original
        # bounds. For example, having a bounds of (-5, 5) we cannot have bounds of (-6, 6)
        # or something else. We are just contracting the bounds INSIDE the original bounds.
        for j, (lower, upper) in enumerate(bounds):
            # Get the best position for this dimension
            best = best_position[j]
            # Update the new bounds for the lower and upper.
            new_lower = best - self._params["memory_coeff"] * (best - lower)
            new_upper = best + self._params["memory_coeff"] * (upper - best)
            # Append the new bounds
            new_bounds.append((new_lower, new_upper))
        # Depending on the flag, return the bounds as a tuple or as a list
        return new_bounds[0] if was_tuple else new_bounds

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
        best_position=np.zeros(dimension),
        stagnation_counter=0
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


def _calc_global_density(particles: list[Particle]) -> float:
    """Compute the global density of the swarm based on the occupied volume."""
    positions = np.array([p.position for p in particles])
    min_vals = np.min(positions, axis=0)
    max_vals = np.max(positions, axis=0)
    volume = np.prod(max_vals - min_vals)  # Compute hypervolume
    # Avoid division by zero
    return len(particles) / volume if volume > 0 else 1.0
