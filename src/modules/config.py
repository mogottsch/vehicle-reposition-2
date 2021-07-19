import os

# Path to unpacked trips folder
PATH_DIR_TRIPS_RAW = os.path.join(
    "/home", "moritz", "data", "Uni", "Bachelorseminar", "Trips"
)


# --- Model Configuration ---

H3_RESOLUTION = 7
H3_RESOLUTION_DOWNSCALING_QUANTILES = [0.9]
PERIOD_DURATION = 8  # in hours
RELOCATION_PERIODS_INDEX = [0]
MODE_IS_WEEKEND = True
N_REALIZATIONS = 3

# the number of scenarios is determined by the period duration and the number of different
# realizations of each random variable
# N_SCENARIOS = N_REALIZATIONS * (|T| - 1)
# where |T| is the number of periods, which is equal to 24/PERIDOD_DURATION, because we
# examining a 24 hour time interval
N_SCENARIOS = int(N_REALIZATIONS ** (24 / PERIOD_DURATION - 1))
N_REDUCED_SCNEARIOS = 4

ALL_VEHICLE_TYPES = ["kick_scooter", "bicycle", "car"]  # short trips -> long trips
VEHICLE_PROFIT = {  # euro / min
    "kick_scooter": 0.19,
    "bicycle": 0.1 / 3,
    "car": 0.29,
}

# relocations at once, e.g. 40 kick scooters in 1 truck
# used for relocation cost calculation
VEHICLE_STACK_RELOCATIONS = {
    "kick_scooter": 40,
    "bicycle": 20,
    "car": 1,
}
RELOCATION_DRIVER_SALARY = 14  # euro/hour

VEHICLE_PARKING_COSTS = {  # eur/2h
    "kick_scooter": 0.05,
    "car": 2,
    "bicycle": 0.1,
}


# --- Solver options ---

SOLVER = "CPLEX_PY"
SOLVER_PATHS = {
    "COIN_CMD": "/usr/bin/cbc",
    "CPLEX_CMD": "/opt/ibm/ILOG/CPLEX_Studio201/cplex/bin/x86-64_linux/cplex",
}
SOLVER_OPTIONS = {
    "logPath": "log",
    "msg": 0,
    # "threads": 16
}

# --- Filepaths ---

ROOT = os.path.abspath(os.path.join("..", ".."))

PATH_HEXAGON_RESOLUTION_MAP = os.path.join(ROOT, "data", "hexagon_resolution_map.pkl")

PATH_TRIPS = os.path.join(ROOT, "data", "trips.pkl")
PATH_TRIPS_GROUPED = os.path.join(ROOT, "data", "trips_grouped.pkl")
PATH_SCENARIOS = os.path.join(ROOT, "data", "scenarios.pkl")
PATH_SCENARIOS_REDUCED = os.path.join(ROOT, "data", "scenarios_reduced.pkl")
PATH_SCENARIO_PROBABILITY = os.path.join(ROOT, "data", "scenarios_probability.pkl")

PATH_SPEEDS = os.path.join(ROOT, "data", "speeds.pkl")
PATH_DISTANCES = os.path.join(ROOT, "data", "distances.pkl")
PATH_FLEET_SIZE = os.path.join(ROOT, "data", "fleet_size.pkl")
PATH_INITIAL_ALLOCATION = os.path.join(ROOT, "data", "initial_allocation.pkl")

PATH_SCENARIO_TREE_NODES = os.path.join(ROOT, "data", "scenario_tree_nodes.pkl")

PATH_RESULTS_SUMMARY = os.path.join(ROOT, "data", "results_summary.pkl")
PATH_RESULTS_VAR_REGION = os.path.join(ROOT, "data", "result_variables_by_region.pkl")
PATH_RESULTS_VAR_TUPLE = os.path.join(ROOT, "data", "result_variables_by_tuple.pkl")
PATH_RESULTS_VALUE_STOCHASTIC = os.path.join(ROOT, "data", "results_1.pkl")
PATH_RESULTS_SINGLE_MODAL_BENCHMARK = os.path.join(ROOT, "data", "results_2.pkl")
PATH_RESULTS_VALUE_AT_RISK = os.path.join(ROOT, "data", "value_at_risk.pkl")

PATH_DIR_FIGURES = os.path.join(ROOT, "figures")
