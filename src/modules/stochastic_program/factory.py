from pandas import DataFrame
import pandas as pd
from pandas.core.series import Series

from modules.measure_time_trait import MeasureTimeTrait
from modules.config import (
    ALL_VEHICLE_TYPES,
    PERIOD_DURATION,
    RELOCATION_PERIODS_INDEX,
)
from modules.stochastic_program.stochastic_program import StochasticProgram


class StochasticProgramFactory(MeasureTimeTrait):
    """
    The Stochastic Program Factory transforms the input data for a stochastic
    program from dataframes to dictionaries and also exctracts metadata.
    It can create stochastic an arbitrary amount of stochastic programs, passing
    the transformed data to each of them.

    Attributes
    ----------
    parameters_ready : bool
        Indicactes whether the input data was transformed (read only)
    initial_allocation_ready : bool
        Indicates whether the initial allocation was set, which is
        necessary to create a stochastic program (read only)

    Methods
    -------
    set_initial_allocation(fleet_capacity)
        Takes the fleet capacity of each vehicle type and creates an
        initial allocation of vehicles. At the moment vehicles get
        distributed equally between all regions.
    create_stochastic_program()
        Creates and returns a new stochastic program with the
        transformed input data. Note that set_initial_allocation
        has to be called before.
    """

    _scenarios: DataFrame
    _distances: DataFrame
    _probabilities: DataFrame
    _initial_allocation: DataFrame
    _node_df: DataFrame

    _demand: dict
    _costs: dict
    _profits: dict
    _weighting: dict
    _node_groups: list

    _regions: list
    _periods: list
    _vehicle_types: list
    _fleet_capacity: dict
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
        self._scenarios = scenarios
        self._distances = distances
        self._probabilities = probabilities
        self._node_df = node_df

        self._demand = {}

        self._costs = {}
        self._profits = {}
        self._weighting = {}

        self._regions = list(
            self._scenarios.index.get_level_values("start_hex_ids").unique()
        )

        _periods = list(
            map(
                lambda time: time.hour,
                self._scenarios.index.get_level_values("time").unique(),
            )
        )
        # add additional period for last vehicle state
        self._periods = _periods + [_periods[-1] + PERIOD_DURATION]

        self._vehicle_types = vehicle_types

        self.include_methods = include_methods

        self.initial_allocation_ready = False
        self.parameters_ready = False

        self._convert_parameters()
        self._set_max_demand()

    def _set_max_demand(self):
        demand_per_region: Series = (
            self._scenarios.reset_index(
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
        profits = self._distances[
            ["profit_kick_scooter", "profit_car", "profit_bicycle"]
        ]
        self._profits = (
            profits.rename(
                columns={
                    "profit_kick_scooter": "kick_scooter",
                    "profit_car": "car",
                    "profit_bicycle": "bicycle",
                }
            )
            .stack()
            .to_dict()
        )

        costs = self._distances[["cost_kick_scooter", "cost_car", "cost_bicycle"]]
        self._costs = (
            costs.rename(
                columns={
                    "cost_kick_scooter": "kick_scooter",
                    "cost_car": "car",
                    "cost_bicycle": "bicycle",
                }
            )
            .stack()
            .to_dict()
        )

    def _convert_demand(self):
        _scenarios = self._scenarios.reset_index()
        _scenarios["time"] = _scenarios["time"].apply(lambda time: time.hour)
        self._demand = _scenarios.set_index(
            ["start_hex_ids", "end_hex_ids", "time", "vehicle_types", "scenarios"]
        )["demand"].to_dict()

    def _convert_probabilities(self):
        self._weighting = self._probabilities["probability"].to_dict()

    def _convert_nodes(self):
        self._node_groups = []
        for _, group in self._node_df.reset_index().groupby("node"):
            self._node_groups.append(
                {
                    "scenarios": list(group.scenarios),
                    "time": list(group.time)[0].hour,
                }
            )

    def set_initial_allocation(
        self,
        fleet_capacity: dict,
    ):
        """Creates and returns a new stochastic program with the
        transformed input data. Note that set_initial_allocation
        has to be called before.
        """
        self._fleet_capacity = fleet_capacity

        n_regions = len(self._regions)

        _initial_allocation = pd.DataFrame(
            index=pd.Index(self._regions, name="hex_ids")
        )
        for vehicle_type in self._vehicle_types:
            allocation_per_hex = int(fleet_capacity[vehicle_type] / n_regions)
            rest = fleet_capacity[vehicle_type] % n_regions

            _initial_allocation[vehicle_type] = [allocation_per_hex] * n_regions

            increment_rest_selector = (_initial_allocation.index[:rest], vehicle_type)

            _initial_allocation.loc[increment_rest_selector] = (
                _initial_allocation.loc[increment_rest_selector] + 1
            )

        self._initial_allocation = _initial_allocation
        self.initial_allocation_ready = True

    def create_stochastic_program(self) -> StochasticProgram:
        if not self.parameters_ready:
            raise Exception("Parameters not set. Run convert_parameters() first.")

        if not self.parameters_ready:
            raise Exception(
                "Initial allocation not set. Run set_initial_allocation() first."
            )

        return StochasticProgram(
            self._demand,
            self._costs,
            self._profits,
            self._weighting,
            self._initial_allocation.to_dict(orient="index"),
            self._node_groups,
            self._regions,
            self._periods,
            self._vehicle_types,
            n_scenarios=self._scenarios.index.get_level_values("scenarios").nunique(),
            fleet_capacity=self._fleet_capacity,
            max_demand=self._max_demand,
            relocation_periods=list(
                map(lambda x: x * PERIOD_DURATION, RELOCATION_PERIODS_INDEX)
            ),
        )
