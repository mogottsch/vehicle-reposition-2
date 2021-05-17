import os

PATH_DIR_TRIPS_RAW = os.path.join(
    "/home", "moritz", "data", "Uni", "Bachelorseminar", "Trips"
)  # insert Path to unpacked Trips folder

PATH_TRIPS = os.path.join("..", "..", "data", "trips.pkl")
PATH_TRIPS_GROUPED = os.path.join("..", "..", "data", "trips_grouped.pkl")
PATH_SCENARIOS = os.path.join("..", "..", "data", "scenarios.pkl")
PATH_SCENARIOS_REDUCED = os.path.join("..", "..", "data", "scenarios_reduced.pkl")
PATH_SCENARIO_PROBABILITY = os.path.join(
    "..", "..", "data", "scenarios_probability.pkl"
)
PATH_DISTANCES = os.path.join("..", "..", "data", "distances.pkl")
PATH_INITIAL_ALLOCATION = os.path.join("..", "..", "data", "initial_allocation.pkl")

H3_RESOLUTION = 6
PERIOD_DURATION = 4  # in hours


N_REALIZATIONS = 4

# the number of scenarios is determined by the period duration and the number of different
# realizations of each random variable
# N_SCENARIOS = N_REALIZATIONS * (|T| - 1)
# where |T| is the number of periods, which is equal to 24/PERIDOD_DURATION, because we
# examining a 24 hour time interval
N_SCENARIOS = int(N_REALIZATIONS ** (24 / PERIOD_DURATION - 1))
N_REDUCED_SCNEARIOS = 8

FLEET_CAPACITY = {
    "kick_scooter": 200,
    "bicycle": 100,
    "car": 50,
}

VEHICLE_ORDERING = ["kick_scooter", "bicycle", "car"]  # short trips -> long trips


# Solver options
SOLVER = "COIN_CMD"
SOLVER_PATHS = {"COIN_CMD": os.path.join("/usr", "bin", "cbc")}
# Solver links
# COIN_CMD: https://github.com/coin-or/Cbc
