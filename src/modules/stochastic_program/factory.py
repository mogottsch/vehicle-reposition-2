from collections import defaultdict
from typing import DefaultDict, List
from pandas import DataFrame

from modules.measure_time_trait import MeasureTimeTrait
from modules.config import (
    N_REDUCED_SCNEARIOS,
    VEHICLE_ORDERING,
)
from modules.stochastic_program.stochastic_program import StochasticProgram


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
