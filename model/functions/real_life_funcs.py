"""
Include the real life functions to test the algorithms, including their name
and possible domain.
"""
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from model.solver import ExperimentFunction


def vrp_objective(
    route: np.ndarray,
    travel_times: np.ndarray,
    deadlines: np.ndarray,
    traffic_noise_std: float
) -> float:
    """Objective function for the VRP problem with stochastic noise in the travel times.

    Args:
        route (np.ndarray): Sequence of cities to visit (indices).
        travel_times (np.ndarray): Matrix of travel times between cities.
        deadlines (np.ndarray): Time deadlines for each city.
        traffic_noise_std (float): Standard deviation of the noise in the travel times.

    Returns:
        float: Total cost of the route.
    """
    total_cost = 0
    current_time = 0

    # Simulate noise in the travel times, this is going to
    # simulate possible traffic jams or other delays.
    noisy_travel_times = travel_times + \
        np.random.normal(0, traffic_noise_std, travel_times.shape)
    # Keep the diagonal as 0. THIS IS A HARD CONSTRAINT FOR THE VRP.
    np.fill_diagonal(noisy_travel_times, 0)

    # Modify the route to round it to the nearest integer.
    route = np.round(route, 0)
    # in case that we've repeated cities in the route, we're going
    # to return an infinite cost.
    # * HARD CONSTRAINT.
    if len(route) != len(set(route)):
        return int(1e6)  # a very large number...

    # Then, iteerate over the calculated route to calculate the total cost.
    for i in range(len(route) - 1):
        # * Add the origin and the destination to know the travel time.
        origin = int(route[i])
        destination = int(route[i+1])
        # Add travel time with noise
        travel_time = noisy_travel_times[origin][destination]
        current_time += travel_time

        # Add a penalty if the current time is greater than the deadline.
        if current_time > deadlines[destination]:
            # Quadratic penalty for lateness.
            # The later the delivery, the higher the cost.
            total_cost += (current_time - deadlines[destination]) ** 2
        total_cost += travel_time  # Add the travel time to the total cost.

    return total_cost


def predictive_maintenance_objective(
    schedule: np.ndarray,
    failure_probs: np.ndarray,
    repair_costs: np.ndarray,
    downtime_costs: float
) -> float:
    """Objective function simulating a predictive maintenance scenario
    with stochastic noise in the failures.

    Each component is a machine.

    Args:
        schedule (np.ndarray): Vector with the maintenance schedule.
        failure_probs (np.ndarray): Failure probabilities for each component.
        repair_costs (np.ndarray): Repair costs for each component.
        downtime_costs (float): Cost per hour of downtime.

    Returns:
        float: Total cost of the maintenance schedule.
    """
    total_cost = 0

    for i, maintenance_time in enumerate(schedule):
        # Probability of failure before maintenance...
        failure_prob = failure_probs[i]

        # * Simulate if the component fails before maintenance.
        # * If it fails, the cost is higher.
        # * If it not fails, the cost is lower.
        if np.random.random() < failure_prob:
            # Unexpected failure, add a downtime cost.
            total_cost += repair_costs[i] + \
                downtime_costs * (maintenance_time - 0)
        else:
            # Preventive maintenance cost.
            # Maintenance costs are lower than preventive.
            total_cost += repair_costs[i] * 0.5

    return total_cost


def chemical_experiment_objective(
    conditions: np.ndarray,
    noise_std: float = 0.1
) -> float:
    """Objective function for optimizing a chemical experiment
    with uncertainty in the measurements.

    #! WARNING: This function only works with a dimension of 3.
    #! Not more, not less.

    Args:
        conditions (np.ndarray): Experimental conditions
                (temperature, concentration, reaction time).
        noise_std (float): Standard deviation of the noise in the measurements.
                Defaults to 0.1.

    Returns:
        float: Negative yield of the reaction. (Minimize)
    """
    if len(conditions) < 3:
        # * This function only works with a dimension of 3.
        # * If it is higher, we're going to take the first 3 values.
        return 0.0
    # Get the temperature, the concentration and the reaction time
    # these are the experimental conditions for the chemical reaction.
    temperature, concentration, reaction_time = conditions[0:3]

    # Theorical yield function (can be modified to fit other experiments)
    theoretical_yield = (
        -((temperature - 300) ** 2) / 500  # Optimal on 300 K
        - ((concentration - 0.5) ** 2) / 0.05  # Optimal on 0.5 M
        - ((reaction_time - 60) ** 2) / 100  # Optimal on 60 minutes
    )
    # * The expected optimal value is suppose to be 0.
    # Add the noise to the theoretical yield
    noisy_yield = theoretical_yield + np.random.normal(0, noise_std)

    # * NOTE: Return the negative yield to minimize the function.
    return -noisy_yield


# ========================================================= #
# DEFINE ALL THE STOCH FUNCTIONS WITH THEIR NAME AND DOMAIN #
real_life_funcs: list["ExperimentFunction"] = [
    {
        "name": "Predictive Maintenance",
        "call": predictive_maintenance_objective,  # type: ignore
        "domain": (0, 50)
    },
    {
        "name": "VRP",
        "call": vrp_objective,  # type: ignore
        "domain": (0, 50)
    },
    {
        "name": "Chemical Experiment",
        "call": chemical_experiment_objective,  # type: ignore
        "domain": (0, 1000)
    }
]  # type: ignore
