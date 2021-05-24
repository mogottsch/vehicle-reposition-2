import datetime
import pandas as pd
from operator import itemgetter
from typing import DefaultDict, List
from pulp import (
    LpMaximize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    LpInteger,
    lpSum,
    value,
    listSolvers,
)
from pulp.apis import getSolver
from pulp.constants import LpBinary, LpContinuous
from pulp.pulp import LpAffineExpression

from modules.measure_time_trait import MeasureTimeTrait
from modules.config import (
    SOLVER_OPTIONS,
    SOLVER_PATHS,
    PERIOD_DURATION,
    SOLVER,
)


class StochasticProgram(MeasureTimeTrait):
    # data
    _demand: DefaultDict
    _costs: DefaultDict
    _profits: DefaultDict
    _weighting: DefaultDict
    _initial_allocation: dict
    _node_groups: List

    # meta data
    _regions: List[str]
    _periods: List[int]
    _vehicle_types: List[str]
    _n_scenarios: int
    _fleet_capacity: dict
    _max_demand: DefaultDict

    # model
    _model: LpProblem
    _Y: dict
    _R: dict
    _X: dict
    _X_: dict
    _V: dict
    _Vb: dict
    _U: dict
    _bigU: dict
    _bigUb: dict
    _eta: LpVariable
    _theta: dict

    _objectives: dict
    _objective_function: LpAffineExpression
    _constraints: dict

    # options
    relocations_disabled: bool
    non_anticipativity_disabled: bool
    _relocation_periods: list

    exclude_methods = [
        "_get_lower_vehicles",
        "_get_R",
        "_get_solver",
        "_get_U",
        "_get_previous_vehicle_type",
    ]

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
        fleet_capacity,
        max_demand,
        relocation_periods,
    ) -> None:
        self._demand = demand
        self._costs = costs
        self._profits = profits
        self._weighting = weighting
        self._initial_allocation = initial_allocation
        self._node_groups = node_groups

        self._regions = regions
        self._periods = periods
        self._vehicle_types = vehicle_types
        self._n_scenarios = n_scenarios
        self._fleet_capacity = fleet_capacity
        self._max_demand = max_demand

        self.relocations_disabled = False
        self.non_anticipativity_disabled = False
        self._relocation_periods = relocation_periods

    # ---------------------------------------------------------------------------- #
    #                               LP Initialization                              #
    # ---------------------------------------------------------------------------- #

    # beta is the weight assigned to the value-at-risk
    # if beta is zero we do not include the value-at-risk
    def create_model(self, warmStart=False, beta=0):
        self._model = LpProblem(name="vehicle-reposition", sense=LpMaximize)

        if not warmStart:
            self._create_variables(beta)

        self._create_objectives()

        self._create_constraints(beta)

        for constraint_group in self._constraints.values():
            for constraint in constraint_group:
                self._model += constraint

        self._model += self._objective_function

    def _create_variables(self, beta) -> None:
        tms = (
            self._periods[:-1],
            self._vehicle_types,
            range(self._n_scenarios),
        )

        itms = (self._regions, *tms)

        ijtms = (self._regions, *itms)

        itmsFull = (
            self._regions,
            self._periods,
            self._vehicle_types,
            range(self._n_scenarios),
        )

        # trips
        self._Y = LpVariable.dicts("y", ijtms, lowBound=0, upBound=None, cat=LpInteger)

        # relocations
        if not self.relocations_disabled:
            filtered_ijtms = (
                self._regions,
                self._regions,
                self._relocation_periods,
                self._vehicle_types,
                range(self._n_scenarios),
            )
            self._R = LpVariable.dicts(
                "r", filtered_ijtms, lowBound=0, upBound=None, cat=LpInteger
            )

        # vehicle system state
        self._X = LpVariable.dicts(
            "x", itmsFull, lowBound=0, upBound=None, cat=LpInteger
        )
        self._X_ = LpVariable.dicts(
            "x'", itmsFull, lowBound=0, upBound=None, cat=LpInteger
        )

        self._V = LpVariable.dicts(
            "v", itmsFull, lowBound=0, upBound=None, cat=LpInteger
        )

        # binary variable - no vehicles remain in region
        self._Vb = LpVariable.dicts("vb", itms, lowBound=0, upBound=1, cat=LpInteger)

        # unfulfilled demand for region tuple
        self._U = LpVariable.dicts("u", ijtms, lowBound=0, upBound=None, cat=LpInteger)

        # unfullfilled demand for region
        self._bigU = LpVariable.dicts(
            "U", itms, lowBound=0, upBound=None, cat=LpInteger
        )

        # binary variable - no unfullfilled demand in regions
        self._bigUb = LpVariable.dicts("Ub", itms, cat=LpBinary)

        if beta != 0:
            self._eta = LpVariable("value-at-risk", cat=LpContinuous)
            self._theta = LpVariable.dicts(
                "theta", range(self._n_scenarios), cat=LpBinary
            )

    # returns an array of vehicle types that can fulfill demand the demand of the
    # specified vehicle type, excluding itself
    # e.g. if vehicle_type = 'car' would return ['kick_scooter', 'bicycle']
    def _get_lower_vehicles(self, vehicle_type):
        return self._vehicle_types[: self._vehicle_types.index(vehicle_type)]

    def _get_previous_vehicle_type(self, vehicle_type):
        prev_index = self._vehicle_types.index(vehicle_type) - 1
        if prev_index == -1:
            return None
        return self._vehicle_types[prev_index]

    def _get_R(self, i, j, t, m, s):
        if not self.relocations_disabled and t in self._relocation_periods:
            return self._R[i][j][t][m][s]
        return 0

    def _get_U(self, i, j, t, m, s):
        if m is None:
            return None
        return self._U[i][j][t][m][s]

    def _create_objectives(self):
        self._objectives = {
            s: lpSum(
                [
                    (
                        self._Y[i][j][t][m][s] * self._profits[(i, j, m)]
                        - self._get_R(i, j, t, m, s) * self._costs[(i, j, m)]
                    )
                    for i in self._regions
                    for j in self._regions
                    for t in self._periods[:-1]
                    for m in self._vehicle_types
                ]
            )
            for s in range(self._n_scenarios)
        }

        self._objective_function = lpSum(
            [
                self._weighting[s] * objective
                for s, objective in self._objectives.items()
            ]
        )

    def _create_constraints(self, beta):
        self._constraints = {}
        self._create_demand_constraints()
        self._create_relocation_binary_constraints()
        self._create_big_u_sum_constraints()
        self._create_unfulfilled_demand_binary_constraints()
        self._create_no_refused_demand_constraints()
        self._create_relocations_constraints()
        self._create_vehicle_trips_starting_constraints()
        self._create_vehicle_trips_ending_constraints()
        self._create_initial_allocation_constraints()

        if not self.non_anticipativity_disabled:
            self._create_non_anticipativity_constraints()

        if beta != 0:
            self._create_value_at_risk_constraints()

    def _create_demand_constraints(self):
        demand_constraints = [
            (
                (
                    self._Y[i][j][t][m][s]
                    == self._demand[(i, j, t, m, s)]
                    - self._U[i][j][t][m][s]
                    + self._get_U(
                        i,
                        j,
                        t,
                        self._get_previous_vehicle_type(m),
                        s,
                    )
                ),
                f"number of trips with {m} from {i} to {j} in period {t} in scenario {s} matches"
                + "demand and unfulfilled demand",
            )
            for i in self._regions
            for j in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]

        self._constraints["demand"] = demand_constraints

    def _create_big_u_sum_constraints(self):
        big_u_sum_constraints = [
            (
                (
                    self._bigU[i][t][m][s]
                    == lpSum([self._U[i][j][t][m][s]] for j in self._regions)
                ),
                f"accumulated unfulfilled trips in region {i} in period {t} with vehicle {m}"
                + f" in scenario {s} is equal to total unfulfilled trips in that region",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints["big_u_sum"] = big_u_sum_constraints

    def _create_relocation_binary_constraints(self):
        relocation_binary_constraints = [
            (
                (self._V[i][t][m][s] <= self._fleet_capacity[m] * self._Vb[i][t][m][s]),
                f"force binary variable Vb to represent whether any vehicles are idle"
                + f"in region {i} in period {t} with vehicle {m} in scenario {s}",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints["relocation_binary"] = relocation_binary_constraints

    def _create_unfulfilled_demand_binary_constraints(self):
        unfulfilled_demand_binary_constraints = [
            (
                (
                    self._bigU[i][t][m][s]
                    <= sum(
                        [
                            self._max_demand[(i, t, m_, s)]
                            for m_ in self._get_lower_vehicles(m) + [m]
                        ]
                    )
                    * self._bigUb[i][t][m][s]
                ),
                f"force binary variable bigUb to represent whether any demand remains unfilled"
                + f"in region {i} in period {t} with vehicle {m} in scenario {s}",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints[
            "unfulfilled_demand_binary"
        ] = unfulfilled_demand_binary_constraints

    def _create_no_refused_demand_constraints(self):
        no_refused_demand_constraints = [
            (
                (self._bigUb[i][t][m][s] + self._Vb[i][t][m][s] <= 1),
                f"only allow to refuse demand if demand cannot be fulfilled due to lack of vehicles"
                + f"in region {i} at period {t} with {m} in scenario {s}",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints["no_refused_demand"] = no_refused_demand_constraints

    def _create_relocations_constraints(self):
        relocations_constraints = [
            (
                (
                    self._X[i][t][m][s]
                    + lpSum(
                        [
                            self._get_R(j, i, t, m, s) - self._get_R(i, j, t, m, s)
                            for j in self._regions
                            if j != i
                        ]
                    )
                    == self._X_[i][t + PERIOD_DURATION][m][s]
                ),
                f"Number of {m} after realized trips in region {i} (period {t}, scenario {s})"
                + f" is equal to the number of vehicles before realized trips of the next period {t +PERIOD_DURATION}"
                + " minus vehicles relocated away plus vehicle relocated into",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]

        self._constraints["relocations_constraints"] = relocations_constraints

    def _create_vehicle_trips_starting_constraints(self):
        vehicle_trips_starting_constraints = [
            (
                self._X_[i][t][m][s]
                == self._V[i][t][m][s]
                + lpSum([self._Y[i][j][t][m][s] for j in self._regions]),
                f"number of {m} in region {i} before demand realization (period {t}, scenario {s})"
                + " is equal to the number of outgoing trips plus the number of idle vehicles",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints[
            "vehicle_trips_starting_constraints"
        ] = vehicle_trips_starting_constraints

    def _create_vehicle_trips_ending_constraints(self):
        vehicle_trips_ending_constraints = [
            (
                self._X[i][t][m][s]
                == self._V[i][t][m][s]
                + lpSum([self._Y[j][i][t][m][s] for j in self._regions]),
                f"number of {m} in region {i} (period {t+1}, scenario {s}) after demand realization"
                + "is equal to the number of idle vehicles plus the number of incoming trips",
            )
            for i in self._regions
            for t in self._periods[:-1]
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints[
            "vehicle_trips_ending_constraints"
        ] = vehicle_trips_ending_constraints

    def _create_initial_allocation_constraints(self):
        initial_allocation_constraints = [
            (
                self._X_[i][0][m][s] == self._initial_allocation[i][m],
                f"starting allocation of {m} in region {i} in scenario {s}",
            )
            for i in self._regions
            for m in self._vehicle_types
            for s in range(self._n_scenarios)
        ]
        self._constraints["initial_allocation"] = initial_allocation_constraints

    def _create_non_anticipativity_constraints(self):
        non_anticipativity_constraints = []
        for node_group in self._node_groups:
            scenarios, time = itemgetter("scenarios", "time")(node_group)

            non_anticipativity_constraints += [
                (
                    self._Y[i][j][time][m][s] == self._Y[i][j][time][m][s_],
                    f"trips from {i} to {j} in period {time} with {m} must"
                    + f" be the same for scenario {s} and {s_}",
                )
                for s, s_ in zip(scenarios[:], scenarios[1:])
                for i in self._regions
                for j in self._regions
                for m in self._vehicle_types
            ]

            if not self.relocations_disabled:
                non_anticipativity_constraints += [
                    (
                        self._get_R(i, j, time, m, s) == self._get_R(i, j, time, m, s_),
                        f"relocations from {i} to {j} in period {time}"
                        + f" with {m} must be the same for scenario {s} and {s_}",
                    )
                    for s, s_ in zip(scenarios[:], scenarios[1:])
                    for i in self._regions
                    for j in self._regions
                    for m in self._vehicle_types
                    if i != j
                ]

            self._constraints["non_anticipativity"] = non_anticipativity_constraints

    def _create_value_at_risk_constraints(self):
        value_at_risk_contraints = []
        [self.eta - self._objective_function for s in range(self._n_scenarios)]

    # ---------------------------------------------------------------------------- #
    #                                 Solving                                      #
    # ---------------------------------------------------------------------------- #

    def _get_solver(self, **kwargs):
        if SOLVER not in listSolvers():
            raise Exception(
                f"{SOLVER} is not supported by PuLP."
                + f" Select one of the following:\n {listSolvers()}"
            )

        if SOLVER in listSolvers(onlyAvailable=True):
            return getSolver(SOLVER, **kwargs)
        if SOLVER in SOLVER_PATHS:
            return getSolver(SOLVER, path=SOLVER_PATHS[SOLVER])
        raise Exception(
            f"{SOLVER} is not available. "
            + "Please install and enter correct path in config or use other solver.\n"
            + f"Available solvers are: {listSolvers(onlyAvailable=True)}"
        )

    def solve(self, **kwargs):
        self._model.solve(solver=self._get_solver(**SOLVER_OPTIONS, **kwargs))
        print("Status:", LpStatus[self._model.status])
        print("Optimal Value of Objective Function: ", value(self._model.objective))

    # ---------------------------------------------------------------------------- #
    #                               Result Evaluation                              #
    # ---------------------------------------------------------------------------- #

    def get_results_by_tuple_df(self):
        results = pd.DataFrame.from_dict(
            {
                (i, j, t, m, s): {
                    "relocations": int(value(self._get_R(i, j, t, m, s))),
                    "trips": int(value(self._Y[i][j][t][m][s])),
                    "accumulated_unfulfilled_demand": int(
                        value(self._U[i][j][t][m][s])
                    ),
                    "demand": self._demand[(i, j, t, m, s)],
                }
                for i in self._regions
                for j in self._regions
                for t in self._periods[:-1]
                for m in self._vehicle_types
                for s in range(self._n_scenarios)
            },
            orient="index",
        )

        results.index = results.index.set_names(
            ["start_hex_ids", "end_hex_ids", "time", "vehicle_types", "scenarios"]
        )
        return results

    def get_results_by_region_df(self):
        results = pd.DataFrame.from_dict(
            {
                (i, t, m, s): {
                    "accumulated_unfulfilled_demand": int(
                        value(self._bigU[i][t][m][s])
                    ),
                    "has_unfulfilled_demand": int(value(self._bigUb[i][t][m][s])),
                    "has_remaining_vehicles": int(value(self._Vb[i][t][m][s])),
                    "n_parking": int(value(self._V[i][t][m][s])),
                    "n_vehicles_after_demand": int(value(self._X[i][t][m][s])),
                    "n_vehicles_before_demand": int(value(self._X_[i][t][m][s])),
                }
                for i in self._regions
                for t in self._periods[:-1]
                for m in self._vehicle_types
                for s in range(self._n_scenarios)
            },
            orient="index",
        )

        results.index = results.index.set_names(
            ["hex_ids", "time", "vehicle_types", "scenario"]
        )
        return results

    def get_summary(self):
        tuple_df = self.get_results_by_tuple_df().reset_index()
        region_df = self.get_results_by_region_df().reset_index()

        return {
            "status": LpStatus[self._model.status],
            "objective": value(self._model.objective),
            # the values belows are summed up for all possible scenarios
            "n_trips_avg": tuple_df["trips"].sum() / self._n_scenarios,
            "n_unfilled_demand_avg": (tuple_df["demand"] - tuple_df["trips"]).sum()
            / self._n_scenarios,
            "demand_avg": tuple_df["demand"].sum() / self._n_scenarios,
            "n_parking_avg": region_df["n_parking"].sum() / self._n_scenarios,
            "n_relocations_avg": tuple_df["relocations"].sum() / self._n_scenarios,
        }

    def get_unfulfilled_demand(self):
        unfulfilled_demand = (
            self.get_results_by_tuple_df()["accumulated_unfulfilled_demand"]
            .to_frame()
            .reset_index()
        )

        unfulfilled_demand["time"] = unfulfilled_demand["time"].map(
            lambda hour: datetime.time(hour=hour)
        )
        unfulfilled_demand = unfulfilled_demand.set_index(
            ["start_hex_ids", "end_hex_ids", "time", "vehicle_types", "scenarios"]
        ).reorder_levels(
            ["scenarios", "start_hex_ids", "end_hex_ids", "time", "vehicle_types"]
        )
        return unfulfilled_demand
