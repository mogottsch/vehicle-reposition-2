from collections import defaultdict
from typing import DefaultDict, List
from pandas import DataFrame
import pandas as pd
from pandas.core.series import Series

from modules.measure_time_trait import MeasureTimeTrait
from modules.config import (
    ALL_VEHICLE_TYPES,
    PERIOD_DURATION,
)
from modules.stochastic_program.stochastic_program import StochasticProgram


class StochasticProgramFactory(MeasureTimeTrait):
    scenarios: DataFrame
    distances: DataFrame
    probabilities: DataFrame
    initial_allocation: DataFrame
    node_df: DataFrame

    demand: dict
    costs: dict
    profits: dict
    weighting: dict
    node_groups: List

    regions: list
    periods: list
    vehicle_types: list
    fleet_capacity: dict
    _max_demand: dict = {}

    parameters_ready: bool
    initial_allocation_ready: bool

    def __init__(
        self,
        scenarios: DataFrame,
        distances: DataFrame,
        probabilities: DataFrame,
        node_df: DataFrame,
        vehicle_types: list = ALL_VEHICLE_TYPES,
        include_methods: list = [],
    ) -> None:
        self.scenarios = scenarios
        self.distances = distances
        self.probabilities = probabilities
        self.node_df = node_df

        self.demand = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )

        self.costs = defaultdict(lambda: defaultdict(dict))
        self.profits = defaultdict(lambda: defaultdict(dict))
        self.weighting = defaultdict()

        self.regions = list(
            self.scenarios.index.get_level_values("start_hex_ids").unique()
        )

        periods = list(
            map(
                lambda time: time.hour,
                self.scenarios.index.get_level_values("time").unique(),
            )
        )
        # add additional period for last vehicle state
        self.periods = periods + [periods[-1] + PERIOD_DURATION]

        self.vehicle_types = vehicle_types

        self.include_methods = include_methods

        self.initial_allocation_ready = False
        self.parameters_ready = False

        self._convert_parameters()
        self._set_max_demand()

    def _set_max_demand(self):
        demand_per_region: Series = (
            self.scenarios.reset_index(
                ["start_hex_ids", "scenarios", "time", "vehicle_types"]
            )
            .groupby(["start_hex_ids", "time", "vehicle_types", "scenarios"])["demand"]
            .sum()
        )
        demand_per_region.index = demand_per_region.index.set_levels(
            demand_per_region.index.levels[1].map(lambda time: time.hour),
            level="time",
            verify_integrity=False,
        )

        self._max_demand = demand_per_region.to_dict()

    def _convert_parameters(self):
        self._convert_probabilities()
        self._convert_distances()
        self._convert_demand()
        self._convert_nodes()

        self.parameters_ready = True

    def _convert_distances(self):
        for _, row in self.distances.reset_index().iterrows():
            for vehicle_type in self.vehicle_types:
                self.costs[row.start_hex_id][row.end_hex_id][vehicle_type] = row[
                    f"cost_{vehicle_type}"
                ]
                self.profits[row.start_hex_id][row.end_hex_id][vehicle_type] = row[
                    f"profit_{vehicle_type}"
                ]

    def _convert_demand(self):
        for _, row in self.scenarios.reset_index().iterrows():
            self.demand[row.start_hex_ids][row.end_hex_ids][row.time.hour][
                row.vehicle_types
            ][row.scenarios] = row.demand

    def _convert_probabilities(self):
        for _, row in self.probabilities.reset_index().iterrows():
            self.weighting[row.scenarios] = row.probability

    def _convert_nodes(self):
        self.node_groups = []
        for _, group in self.node_df.reset_index().groupby("node"):
            self.node_groups.append(
                {
                    "scenarios": list(group.scenarios),
                    "time": list(group.time)[0].hour,
                }
            )

    def set_initial_allocation(
        self,
        fleet_capacity: dict,
    ):
        self.fleet_capacity = fleet_capacity

        n_regions = len(self.regions)

        initial_allocation = pd.DataFrame(index=pd.Index(self.regions, name="hex_ids"))
        for vehicle_type in self.vehicle_types:
            allocation_per_hex = int(fleet_capacity[vehicle_type] / n_regions)
            rest = fleet_capacity[vehicle_type] % n_regions

            initial_allocation[vehicle_type] = [allocation_per_hex] * n_regions

            increment_rest_selector = (initial_allocation.index[:rest], vehicle_type)

            initial_allocation.loc[increment_rest_selector] = (
                initial_allocation.loc[increment_rest_selector] + 1
            )

        self.initial_allocation = initial_allocation
        self.initial_allocation_ready = True

    def create_stochastic_program(self) -> StochasticProgram:
        if not self.parameters_ready:
            raise Exception("Parameters not set. Run convert_parameters() first.")

        if not self.parameters_ready:
            raise Exception(
                "Initial allocation not set. Run set_initial_allocation() first."
            )

        return StochasticProgram(
            self.demand,
            self.costs,
            self.profits,
            self.weighting,
            self.initial_allocation.to_dict(orient="index"),
            self.node_groups,
            self.regions,
            self.periods,
            self.vehicle_types,
            n_scenarios=self.scenarios.index.get_level_values("scenarios").nunique(),
            fleet_capacity=self.fleet_capacity,
            max_demand=self._max_demand,
        )
