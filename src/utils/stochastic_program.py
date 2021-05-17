import pandas as pd
from collections import defaultdict
from operator import itemgetter
from sys import path
from typing import Any, Callable, DefaultDict, List
from timeit import default_timer as timer
from pulp import (
    LpMaximize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    LpInteger,
    lpSum,
    value,
)
from pandas import DataFrame
from pulp.apis.coin_api import COIN_CMD
from pulp.pulp import LpAffineExpression

from utils.config import (
    FLEET_CAPACITY,
    N_REDUCED_SCNEARIOS,
    PATH_TO_COIN_CMD,
    PERIOD_DURATION,
    SOLVER,
    VEHICLE_ORDERING,
)


class MeasureTimeTrait(object):
    exclude_methods: List = []

    def __getattribute__(self, name: str) -> Any:
        attribute = object.__getattribute__(self, name)
        if callable(attribute) and name not in self.exclude_methods:
            return object.__getattribute__(self, "measure_time")(attribute)
        return attribute

    def measure_time(self, cb: Callable) -> Callable:
        def decorated_callable(*args, **kwargs):
            start = timer()
            value = cb(*args, **kwargs)
            end = timer()
            print(f"{cb.__name__} finished in {end-start:.2f} seconds")
            return value

        return decorated_callable


class StochasticProgram(MeasureTimeTrait):
    # data
    demand: DefaultDict
    costs: DefaultDict
    profits: DefaultDict
    weighting: DefaultDict
    initial_allocation: dict
    node_groups: List

    # meta data
    regions: List[str]
    periods: List[int]
    vehicle_types: List[str]
    n_scenarios: int

    M: int

    # model
    model: LpProblem
    Y: dict
    R: dict
    X: dict
    objective_function: LpAffineExpression
    constraints: dict

    # options
    relocations_disabled: bool

    exclude_methods = ["get_lower_vehicles", "get_R"]

    def __init__(
        self,
        demand,
        costs,
        profits,
        weighting,
        initial_allocation,
        node_groups,
        regions,
        periods,
        vehicle_types,
        n_scenarios,
    ) -> None:
        self.demand = demand
        self.costs = costs
        self.profits = profits
        self.weighting = weighting
        self.initial_allocation = initial_allocation
        self.node_groups = node_groups

        self.regions = regions
        self.periods = periods
        self.vehicle_types = vehicle_types

        self.n_scenarios = n_scenarios

        self.M = 100_000

        self.relocations_disabled = False

    def get_solver(self, **kwargs):
        if SOLVER == "COIN_CMD":
            return COIN_CMD(**kwargs)

        raise Exception("Invalid Solver Config")

    def solve(self, **kwargs):
        self.model.solve(solver=self.get_solver(**kwargs))
        print("Status:", LpStatus[self.model.status])
        print("Optimal Value of Objective Function: ", value(self.model.objective))
        print(f"Runtime without preprocessing: {self.model.solutionTime:.2f} seconds")

    def get_results_by_tuple_df(self):
        results = pd.DataFrame.from_dict(
            {
                (i, j, t, m, s): {
                    "relocations/parking": int(value(self.get_R(i, j, t, m, s))),
                    "trips": int(value(self.Y[i][j][t][m][s])),
                    "unfulfilled_demand": int(value(self.U[i][j][t][m][s])),
                }
                for i in self.regions
                for j in self.regions
                for t in self.periods[:-1]
                for m in self.vehicle_types
                for s in range(self.n_scenarios)
            },
            orient="index",
        )

        results.index = results.index.set_names(
            ["start_hex_ids", "end_hex_ids", "time", "vehicle_types", "scenario"]
        )
        return results

    def get_results_by_region_df(self):
        results = pd.DataFrame.from_dict(
            {
                (i, t, m, s): {
                    "accumulated_unfulfilled_demand": int(value(self.bigU[i][t][m][s])),
                    "has_unfulfilled_demand": int(value(self.bigUb[i][t][m][s])),
                    "has_remaining_vehicles": int(value(self.Rb[i][t][m][s])),
                    "n_vehicles": int(value(self.X[i][t][m][s])),
                }
                for i in self.regions
                for t in self.periods[:-1]
                for m in self.vehicle_types
                for s in range(self.n_scenarios)
            },
            orient="index",
        )

        results.index = results.index.set_names(
            ["hex_ids", "time", "vehicle_types", "scenario"]
        )
        return results

    def create_model(self):
        self.model = LpProblem(name="vehicle-reposition", sense=LpMaximize)

        self.create_variables()

        self.create_objective_function()
        self.model += self.objective_function

        self.create_constraints()

        for constraint_group in self.constraints.values():
            for constraint in constraint_group:
                self.model += constraint

    def create_variables(self) -> None:
        tms = (
            self.periods[:-1],
            self.vehicle_types,
            range(self.n_scenarios),
        )

        itms = (self.regions, *tms)

        ijtms = (self.regions, *itms)

        itmsFull = (
            self.regions,
            self.periods,
            self.vehicle_types,
            range(self.n_scenarios),
        )

        # trips
        self.Y = LpVariable.dicts("y", ijtms, lowBound=0, upBound=None, cat=LpInteger)

        # relocations
        if self.relocations_disabled:
            self.R = LpVariable.dicts(
                "r", itms, lowBound=0, upBound=None, cat=LpInteger
            )
            self.R = {key: {key: value} for key, value in self.R.items()}
        else:
            self.R = LpVariable.dicts(
                "r", ijtms, lowBound=0, upBound=None, cat=LpInteger
            )

        # vehicle system state
        self.X = LpVariable.dicts(
            "x", itmsFull, lowBound=0, upBound=None, cat=LpInteger
        )

        # binary variable - no vehicles remain in region
        self.Rb = LpVariable.dicts("rb", itms, lowBound=0, upBound=1, cat=LpInteger)

        # unfulfilled demand for region tuple
        self.U = LpVariable.dicts("u", ijtms, lowBound=0, upBound=None, cat=LpInteger)

        # unfullfilled demand for region
        self.bigU = LpVariable.dicts("U", itms, lowBound=0, upBound=None, cat=LpInteger)

        # binary variable - no unfullfilled demand in regions
        self.bigUb = LpVariable.dicts("Ub", itms, lowBound=0, upBound=1, cat=LpInteger)

    # returns an array of vehicle types that can fulfill demand the demand of the
    # specified vehicle type, excluding itself
    # e.g. if vehicle_type = 'car' would return ['kick_scooter', 'bicycle']
    def get_lower_vehicles(self, vehicle_type):
        return self.vehicle_types[: self.vehicle_types.index(vehicle_type)]

    def get_R(self, i, j, t, m, s):
        if not self.relocations_disabled or i == j:
            return self.R[i][j][t][m][s]
        return 0

    def create_objective_function(self):
        self.objective_function = lpSum(
            [
                self.weighting[s]
                * (
                    self.Y[i][j][t][m][s] * self.profits[i][j][m]
                    - self.get_R(i, j, t, m, s) * self.costs[i][j][m]
                )
                for i in self.regions
                for j in self.regions
                for t in self.periods[:-1]
                for m in self.vehicle_types
                for s in range(self.n_scenarios)
            ]
        )

    def create_constraints(self):
        self.constraints = {}
        self.create_demand_constraints()
        self.create_big_u_sum_constraints()
        self.create_relocation_binary_constraints()
        self.create_unfulfilled_demand_binary_constraints()
        self.create_no_refused_demand_constraints()
        self.create_max_trips_constraints()
        self.create_vehicle_movement_constraints()
        self.create_initial_allocation_constraints()
        self.create_non_anticipativity_constraints()

    def create_demand_constraints(self):
        demand_constraints = [
            (
                (
                    self.Y[i][j][t][m][s]
                    == self.demand[i][j][t][m][s]
                    - self.U[i][j][t][m][s]
                    + lpSum(
                        [self.U[i][j][t][m_][s] for m_ in self.get_lower_vehicles(m)]
                    )
                ),
                f"number of trips with {m} from {i} to {j} in period {t} in scenario {s} matches"
                + "demand and unfulfilled demand",
            )
            for i in self.regions
            for j in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]

        self.constraints["demand"] = demand_constraints

    def create_big_u_sum_constraints(self):
        big_u_sum_constraints = [
            (
                (
                    self.bigU[i][t][m][s]
                    == lpSum(
                        [self.U[i][j][t][m_][s]]
                        for j in self.regions
                        for m_ in self.get_lower_vehicles(m) + [m]
                    )
                ),
                f"accumulated unfulfilled trips in region {i} in period {t} with vehicle {m}"
                + f" in scenario {s} is equal to total unfulfilled trips in that region",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints["big_u_sum"] = big_u_sum_constraints

    def create_relocation_binary_constraints(self):
        relocation_binary_constraints = [
            (
                (self.R[i][i][t][m][s] <= FLEET_CAPACITY[m] * self.Rb[i][t][m][s]),
                f"force binary variable Rb to represent whether any vehicles are remaining"
                + f"in region {i} in period {t} with vehicle {m} in scenario {s}",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints["relocation_binary"] = relocation_binary_constraints

    def create_unfulfilled_demand_binary_constraints(self):
        unfulfilled_demand_binary_constraints = [
            (
                (self.bigU[i][t][m][s] <= self.M * self.bigUb[i][t][m][s]),
                f"force binary variable bigUb to represent whether any demand remains unfilled"
                + f"in region {i} in period {t} with vehicle {m} in scenario {s}",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints[
            "unfulfilled_demand_binary"
        ] = unfulfilled_demand_binary_constraints

    def create_no_refused_demand_constraints(self):
        no_refused_demand_constraints = [
            (
                (self.bigUb[i][t][m][s] + self.Rb[i][t][m][s] <= 1),
                f"only allow to refuse demand if demand cannot be fulfilled due to lack of vehicles"
                + f"in region {i} at period {t} with {m} in scenario {s}",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints["no_refused_demand"] = no_refused_demand_constraints

    def create_max_trips_constraints(self):
        max_trips_constraints = [
            (
                (
                    lpSum(
                        [
                            self.Y[i][j][t][m][s] + self.get_R(i, j, t, m, s)
                            for j in self.regions
                        ]
                    )
                    == self.X[i][t][m][s]
                ),
                f"maximum trips from {i} in period {t} with vehicle {m} in scenario {s}",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]

        self.constraints["max_trips_constraints"] = max_trips_constraints

    def create_vehicle_movement_constraints(self):
        vehicle_movement_constraints = [
            (
                self.X[i][t + PERIOD_DURATION][m][s]
                == lpSum(
                    [
                        self.Y[j][i][t][m][s]
                        + self.get_R(
                            j,
                            i,
                            t,
                            m,
                            s,
                        )
                        for j in self.regions
                    ]
                ),
                f"number of {m} in {i} in period {t+1} in scenario {s}"
                + " matches trips, relocations and parking vehicles from previous period",
            )
            for i in self.regions
            for t in self.periods[:-1]
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints["vehicle_movement_constraints"] = vehicle_movement_constraints

    def create_initial_allocation_constraints(self):
        initial_allocation_constraints = [
            (
                self.X[i][0][m][s] == self.initial_allocation[i][m],
                f"starting allocation of {m} in region {i} in scenario {s}",
            )
            for i in self.regions
            for m in self.vehicle_types
            for s in range(self.n_scenarios)
        ]
        self.constraints["initial_allocation"] = initial_allocation_constraints

    def create_non_anticipativity_constraints(self):
        non_anticipativity_constraints = []
        for node_group in self.node_groups:
            scenarios, time = itemgetter("scenarios", "time")(node_group)

            non_anticipativity_constraints += [
                (
                    self.Y[i][j][time][m][s] == self.Y[i][j][time][m][s_],
                    f"trips from {i} to {j} in period {time} with {m} must"
                    + f" be the same for scenario {s} and {s_}",
                )
                for s, s_ in zip(scenarios[:], scenarios[1:])
                for i in self.regions
                for j in self.regions
                for m in self.vehicle_types
            ]

            if self.relocations_disabled:
                non_anticipativity_constraints += [
                    (
                        self.get_R(i, i, time, m, s) == self.get_R(i, i, time, m, s_),
                        f"parking vehicles in region {i} in period {time} with {m} must"
                        + f" be the same for scenario {s} and {s_}",
                    )
                    for s, s_ in zip(scenarios[:], scenarios[1:])
                    for i in self.regions
                    for m in self.vehicle_types
                ]
            else:
                non_anticipativity_constraints += [
                    (
                        self.get_R(i, j, time, m, s) == self.get_R(i, j, time, m, s_),
                        f"relocations/parking vehicles from {i} to {j} in period {time}"
                        + f" with {m} must be the same for scenario {s} and {s_}",
                    )
                    for s, s_ in zip(scenarios[:], scenarios[1:])
                    for i in self.regions
                    for j in self.regions
                    for m in self.vehicle_types
                ]

            self.constraints["non_anticipativity"] = non_anticipativity_constraints


class StochasticProgramFactory(MeasureTimeTrait):
    scenarios: DataFrame
    distances: DataFrame
    probabilities: DataFrame
    initial_allocation: DataFrame
    node_df: DataFrame

    demand: DefaultDict
    costs: DefaultDict
    profits: DefaultDict
    weighting: DefaultDict
    node_groups: List

    def __init__(
        self,
        scenarios: DataFrame,
        distances: DataFrame,
        probabilities: DataFrame,
        initial_allocation: DataFrame,
        node_df: DataFrame,
    ) -> None:
        self.scenarios = scenarios
        self.distances = distances
        self.probabilities = probabilities
        self.initial_allocation = initial_allocation
        self.node_df = node_df

        self.demand = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        self.costs = defaultdict(lambda: defaultdict(dict))
        self.profits = defaultdict(lambda: defaultdict(dict))
        self.weighting = defaultdict()

    def convert_parameters(self):
        self.convert_probabilities()
        self.convert_distances()
        self.convert_demand()
        self.convert_nodes()

    def convert_distances(self):
        vehicle_types = list(self.scenarios.reset_index()["vehicle_types"].unique())

        for _, row in self.distances.reset_index().iterrows():
            for vehicle_type in vehicle_types:
                self.costs[row.start_hex_id][row.end_hex_id][vehicle_type] = row[
                    f"cost_{vehicle_type}"
                ]
                self.profits[row.start_hex_id][row.end_hex_id][vehicle_type] = row[
                    f"profit_{vehicle_type}"
                ]

    def convert_demand(self):
        for _, row in self.scenarios.reset_index().iterrows():
            self.demand[row.start_hex_ids][row.end_hex_ids][row.time.hour][
                row.vehicle_types
            ][row.scenarios] = row.demand

    def convert_probabilities(self):
        for _, row in self.probabilities.reset_index().iterrows():
            self.weighting[row.scenarios] = row.probability

    def convert_nodes(self):
        self.node_groups = []
        for _, group in self.node_df.reset_index().groupby("node"):
            self.node_groups.append(
                {
                    "scenarios": list(group.scenarios),
                    "time": list(group.time)[0].hour,
                }
            )

    def create_stochastic_program(self) -> StochasticProgram:
        regions = list(self.scenarios.index.get_level_values("start_hex_ids").unique())
        periods = list(
            map(
                lambda time: time.hour,
                self.scenarios.index.get_level_values("time").unique(),
            )
        )
        vehicle_types = VEHICLE_ORDERING

        return StochasticProgram(
            self.demand,
            self.costs,
            self.profits,
            self.weighting,
            self.initial_allocation.to_dict(orient="index"),
            self.node_groups,
            regions,
            periods,
            vehicle_types,
            n_scenarios=N_REDUCED_SCNEARIOS,
        )
