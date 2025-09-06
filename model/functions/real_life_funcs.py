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
    traffic_noise_std: float,
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
    noisy_travel_times = travel_times + np.random.normal(
        0, traffic_noise_std, travel_times.shape
    )
    # Keep the diagonal as 0. THIS IS A HARD CONSTRAINT FOR THE VRP.
    np.fill_diagonal(noisy_travel_times, 0)

    # Modify the route to round it to the nearest integer.
    route = np.round(route, 0)
    # in case that we've repeated cities in the route, we're going
    # to return an infinite cost.
    # * HARD CONSTRAINT. If the route doesn't have the same length as the deadlines,
    # * we're going to return a very large number to avoid this solution.
    if len(route) != len(set(route)) or len(route) != len(deadlines):
        return float("1e12")  # a very large number...

    # Then, iteerate over the calculated route to calculate the total cost.
    for i in range(len(route) - 1):
        # * Add the origin and the destination to know the travel time.
        origin = int(route[i])
        destination = int(route[i + 1])
        # Add travel time with noise
        travel_time = noisy_travel_times[origin][destination]
        current_time += travel_time

        # # Add a penalty if the current time is greater than the deadline.
        if current_time > deadlines[destination]:
            # The later the delivery, the higher the cost.
            total_cost += current_time - deadlines[destination]
        total_cost += travel_time  # Add the travel time to the total cost.

    # At the end, add one penalty for the last city to return to the depot.
    total_cost += noisy_travel_times[int(route[-1])][int(route[0])]
    return total_cost


def predictive_maintenance_objective(
    schedule: np.ndarray,
    failure_probs: np.ndarray,
    repair_costs: np.ndarray,
    downtime_costs: float,
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
            total_cost += repair_costs[i] + downtime_costs * (maintenance_time - 0)
        else:
            # Preventive maintenance cost.
            # Maintenance costs are lower than preventive.
            total_cost += repair_costs[i] * 0.5

    return total_cost


def chemical_experiment_objective(
    conditions: np.ndarray, noise_std: float = 0.1
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


# Add this constants for the Financial Problem implementation
# Example expected returns vector and covariance matrix for n assets.
# In a real-world scenario, these would be estimated from historical data.
mu = np.array([0.10, 0.12, 0.08, 0.07])  # Expected returns for 4 assets
Sigma = np.array(
    [
        [0.005, -0.010, 0.004, 0.002],
        [-0.010, 0.040, -0.002, 0.003],
        [0.004, -0.002, 0.023, 0.002],
        [0.002, 0.003, 0.002, 0.018],
    ]
)

# Trade-off parameter (lambda) between risk and return.
# A higher lambda puts more weight on return.
LAMBDA_TRADEOFF = 0.5


def financial_portfolio_objective(x: np.ndarray) -> float:
    """Objective function for financial portfolio optimization.

    The function normalizes x so that the weights sum to 1 (if sum > 0)
    and penalizes any negative weights.
    It then computes:

        f(x) = (portfolio variance) - lambda_tradeoff * (portfolio expected return)

    Lower f(x) is better.

    Args:
        x: np.ndarray of shape (n_assets,)
           A candidate solution representing portfolio weights.
    """
    # Enforce non-negativity; penalize negative weights heavily.
    if np.any(x < 0):
        return 1e6 + np.sum(np.abs(x[x < 0]))  # A large penalty

    # Normalize the portfolio weights to sum to one
    total = np.sum(x)
    if total > 0:
        x_norm: np.ndarray = x / total  # type: ignore
    else:
        x_norm = x

    # Calculate portfolio variance (risk)
    variance: np.ndarray = x_norm.T @ Sigma @ x_norm

    # Calculate portfolio expected return
    expected_return = x_norm.T @ mu

    # Define the objective as risk minus lambda_tradeoff times return.
    # (Since we are minimizing the function, a lower value indicates lower risk and higher return.)
    objective_value = variance - LAMBDA_TRADEOFF * expected_return
    return -objective_value  # type: ignore


# Constants for the microgrid energy management problem
DEMAND = 100.0  # Demand required in MW
RENEWABLE = 60.0  # Renewable generation in MW
GRID_COST = 50.0  # Cost per mW of grid usage
BATTERY_COST = 30.0  # Cost for battery usage
PENALTY_MICROGRID = 1000.0  # Penalty factor for unmet demand


def microgrid_objective(x: np.ndarray) -> float:
    """Objective function for energy management in a microgrid.

    Candidate solution x is a vector where:
      x[0]: Power drawn from the grid (MW)
      x[1]: Battery discharge (MW)
      x[2]: Battery charge (MW)

    The net supply is given by:
      supply = renewable_generation + grid_usage + battery_discharge - battery_charge

    We aim to minimize the total cost:
      cost = grid_cost * grid_usage + battery_cost * (battery_discharge + battery_charge)
             + penalty_factor * (unmet_demand)

    If supply < demand, a heavy penalty is added.
    Negative values are penalized para evitar soluciones inviables.
    """
    # Verify that there's no negative values (negative energy usage is not allowed)
    if np.any(x < 0):
        return 1e6 + np.sum(np.abs(x[x < 0]))

    # Calculate the net supply: sum of renewable generation, grid usage, battery discharge,
    # minus battery charge, and penalize if the demand is not met.
    supply = RENEWABLE + x[0] + x[1] - x[2]

    # Penalización si la oferta no satisface la demanda
    # Penalty if the offer doesn't satisfy the demand
    unmet_demand = max(0, DEMAND - supply)
    penalty = PENALTY_MICROGRID * unmet_demand

    # Total cost: sum of costs for energy taken from the grid
    # and battery usage, plus the penalty
    return GRID_COST * x[0] + BATTERY_COST * (x[1] + x[2]) + penalty


# Short functions that represent some real-life applications
def cache_optimization(x: np.ndarray) -> float:
    """Simulates the cost of caching storage with stochastic requests.
    **Reference:** Caching optimization principles can be found in "Efficient Cache Management Policies", SIGCOMM.
    **Global Optimum:** Approximately 0 at x ≈ 1.
    """
    noise = np.random.normal(0, 2, len(x))
    return np.sum(np.exp(-x) + 0.1 * x**2) + np.sum(noise)


def production_scheduling(x: np.ndarray) -> float:
    """Models production scheduling in a factory with uncertain demand and failures.
    **Reference:** Industrial scheduling methods as described in "Production and Operations Management", Elsevier.
    **Global Optimum:** Approximately 0 at x ≈ 50.
    """
    noise = np.random.normal(0, 5, len(x))
    return np.sum((x - 50) ** 2 + np.sin(0.1 * np.pi * x)) + np.sum(noise)


def online_ads_bidding(x: np.ndarray) -> float:
    """Optimizes advertisement bidding under uncertain costs and fluctuating competition.
    **Reference:** Online ad auction strategies found in "Mechanism Design for Online Advertising", Springer.
    **Global Optimum:** Approximately 0 at x ≈ 2.
    """
    noise = np.random.normal(0, 3, len(x))
    return np.sum(x**3 - 3 * x**2 + 2 * x) + np.sum(noise)


def network_packet_routing(x: np.ndarray) -> float:
    """Simulates network packet routing considering stochastic congestion.
    **Reference:** Network traffic optimization principles in "Computer Networks: A Systems Approach", Morgan Kaufmann.
    **Global Optimum:** Approximately 0 at x ≈ 0.
    """
    noise = np.random.normal(0, 4, len(x))
    return np.sum(np.tanh(x) + 0.5 * x**2) + np.sum(noise)


def retail_inventory_optimization(x: np.ndarray) -> float:
    """Models retail inventory management with variable demand and storage costs.
    **Reference:** Inventory control models from "Supply Chain Management: Strategy, Planning, and Operation", Pearson.
    **Global Optimum:** Approximately 0 at x ≈ 100.
    """
    noise = np.random.normal(0, 6, len(x))
    return np.sum(np.abs(x - 100) ** 1.5) + np.sum(noise)


def supply_chain_network_design(x: np.ndarray) -> float:
    """Optimizes supply chain network design considering transportation costs and demand variability.
    **Reference:** Supply chain network design in "Supply Chain Management: Strategy, Planning, and Operation", Pearson.
    **Global Optimum:** Approximately 0 at x ≈ 50.
    """
    noise = np.random.normal(0, 8, len(x))
    return np.sum(np.abs(x - 50) ** 2) + np.sum(noise)


# ========================================================= #
# DEFINE ALL THE STOCH FUNCTIONS WITH THEIR NAME AND DOMAIN #
real_life_funcs: list["ExperimentFunction"] = [
    # {
    #     "name": "Predictive Maintenance",
    #     "call": predictive_maintenance_objective,  # type: ignore
    #     "domain": (0, 29),
    #     "dimension": 30,
    # },
    # {
    #     "name": "VRP",
    #     "call": vrp_objective,  # type: ignore
    #     "domain": (0, 9),
    #     "dimension": 10,
    # },
    # {
    #     "name": "Chemical Experiment",
    #     "call": chemical_experiment_objective,  # type: ignore
    #     "domain": (0, 1000),
    #     "dimension": 3,
    # },
    # {
    #     "name": "Financial Portfolio",
    #     "call": financial_portfolio_objective,
    #     "domain": (0, 1),
    #     "dimension": 4,
    # },
    # {
    #     "name": "Microgrid Energy Management",
    #     "call": microgrid_objective,
    #     "domain": (0, 100),
    #     "dimension": 3,
    # },
    {
        "name": "Cache Optimization",
        "call": cache_optimization,
        "domain": (0, 1),
        # "dimension": 10,
        "optimal_value": 0,
        "optimal_x_value": [1],
    },
    {
        "name": "Production Scheduling",
        "call": production_scheduling,
        "domain": (0, 100),
        # "dimension": 5,
        "optimal_value": 0,
        "optimal_x_value": [50],
    },
    {
        "name": "Online Ads Bidding",
        "call": online_ads_bidding,
        "domain": (0, 10),
        # "dimension": 6,
        "optimal_value": 0,
        "optimal_x_value": [2],
    },
    {
        "name": "Network Packet Routing",
        "call": network_packet_routing,
        "domain": (0, 50),
        # "dimension": 8,
        "optimal_value": 0,
        "optimal_x_value": [0],
    },
    {
        "name": "Retail Inventory Optimization",
        "call": retail_inventory_optimization,
        "domain": (0, 200),
        # "dimension": 7,
        "optimal_value": 0,
        "optimal_x_value": [100],
    },
    {
        "name": "Supply Chain Network Design",
        "call": supply_chain_network_design,
        "domain": (0, 100),
        # "dimension": 6,
        "optimal_value": 0,
        "optimal_x_value": [50],
    },
]  # type: ignore
