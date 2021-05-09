PATH_DIR_TRIPS_RAW = (
    R"D:\Data\Uni\Bachelorseminar\Trips"  # insert Path to unpacked Trips folder
)
PATH_TRIPS = R"..\..\data\trips.pkl"
PATH_TRIPS_GROUPED = R"..\..\data\trips_grouped.pkl"
PATH_SCENARIOS = R"..\..\data\scenarios.pkl"
PATH_SCENARIOS_REDUCED = R"..\..\data\scenarios_reduced.pkl"

H3_RESOLUTION = 6
PERIOD_DURATION = 4  # in hours


N_REALIZATIONS = 4

# the number of scenarios is determined by the period duration and the number of different
# realizations of each random variable
# N_SCENARIOS = N_REALIZATIONS * (|T| - 1)
# where |T| is the number of periods, which is equal to 24/PERIDOD_DURATION, because we
# examining a 24 hour time interval
N_SCENARIOS = int(N_REALIZATIONS ** (24 / PERIOD_DURATION - 1))
N_REDUCED_SCNEARIOS = 32