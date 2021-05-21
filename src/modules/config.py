import os

# insert Path to unpacked trips folder
PATH_DIR_TRIPS_RAW = os.path.join(
    "/home", "moritz", "data", "Uni", "Bachelorseminar", "Trips"
)

H3_RESOLUTION = 7
PERIOD_DURATION = 12  # in hours
RELOCATION_PERIODS_INDEX = [0]
MODE_IS_WEEKEND = True

N_REALIZATIONS = 4

# the number of scenarios is determined by the period duration and the number of different
# realizations of each random variable
# N_SCENARIOS = N_REALIZATIONS * (|T| - 1)
# where |T| is the number of periods, which is equal to 24/PERIDOD_DURATION, because we
# examining a 24 hour time interval
N_SCENARIOS = int(N_REALIZATIONS ** (24 / PERIOD_DURATION - 1))
N_REDUCED_SCNEARIOS = 2

ALL_VEHICLE_TYPES = ["kick_scooter", "bicycle", "car"]  # short trips -> long trips


# Solver options
SOLVER = "COIN_CMD"
SOLVER_PATHS = {"COIN_CMD": os.path.join("/usr", "bin", "cbc")}
SOLVER_OPTIONS = {"logPath": "log", "msg": 0, "threads": 15}
# Solver links
# COIN_CMD: https://github.com/coin-or/Cbc


ROOT = os.path.abspath(os.path.join("..", ".."))

PATH_TRIPS = os.path.join(ROOT, "data", "trips.pkl")
PATH_TRIPS_GROUPED = os.path.join(ROOT, "data", "trips_grouped.pkl")
PATH_SCENARIOS = os.path.join(ROOT, "data", "scenarios.pkl")
PATH_SCENARIOS_REDUCED = os.path.join(ROOT, "data", "scenarios_reduced.pkl")
PATH_SCENARIO_PROBABILITY = os.path.join(ROOT, "data", "scenarios_probability.pkl")

PATH_DISTANCES = os.path.join(ROOT, "data", "distances.pkl")
PATH_FLEET_SIZE = os.path.join(ROOT, "data", "fleet_size.pkl")
PATH_INITIAL_ALLOCATION = os.path.join(ROOT, "data", "initial_allocation.pkl")

PATH_RESULTS_SUMMARY = os.path.join(ROOT, "data", "results_summary.pkl")
PATH_RESULTS_VAR_REGION = os.path.join(ROOT, "data", "result_variables_by_region.pkl")
PATH_RESULTS_VAR_TUPLE = os.path.join(ROOT, "data", "result_variables_by_tuple.pkl")
PATH_RESULTS_VALUE_STOCHASTIC = os.path.join(ROOT, "data", "results_1.pkl")
PATH_RESULTS_SINGLE_MODAL_BENCHMARK = os.path.join(ROOT, "data", "results_2.pkl")

PATH_DIR_FIGURES = os.path.join(ROOT, "figures")
