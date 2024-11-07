"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

from typing import Type, Union
import numpy as np
import numpy.typing as npt


class Activity:
    """
    Describes a human activity
    
    Properties:
        name (str): name of human activity
        id (int): identification number of activity
        change (float): change of human activity, greater than 0
        uncertainty (float): uncertainty of change
    """
    def __init__(self, name: str, id: int):
        self.name = name
        self._id =  id
        self._change = 1.0
        self._uncertainty = 0.0

    def __repr__(self) -> str:
        return f"Activity({self.id}, {self.name})"

    @property
    def id(self) -> int:
        return self._id

    @property
    def change(self) -> float:
        return self._change

    @change.setter
    def change(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Value has to be greater than 0")
        self._change = value

    @property
    def uncertainty(self) -> float:
        return self._uncertainty
    
    @uncertainty.setter
    def uncertainty(self, value: float) -> None:
        self._uncertainty = value


class Pressure:
    """
    Pressures describe the effects of human activities and protective measures

    Properties:
        name (str): name of pressure
        id (int): identification number of pressure
        pressure (float): baseline pressure value, default is 1.0
        expected_pressure (float): pressure value based on human activities and protective measures
        uncertainty (float): uncertainty of expected_pressure
    """
    def __init__(self, name: str, id: int):
        self.name = name
        self._id = id
        self._pressure = 1.0
        self._pressure_reduction = 0.0
        self._uncertainty = 0.0

    def __repr__(self) -> str:
        return f"Pressure({self.id}, {self.name})"

    @property
    def id(self) -> int:
        return self._id

    @property
    def pressure(self) -> float:
        return self._pressure
    
    @pressure.setter
    def pressure(self, value: float) -> None:
        self._pressure = value

    @property
    def expected_pressure(self) -> float:
        value = self._pressure - self._pressure_reduction 
        return value
    
    @property
    def pressure_reduction(self) -> float:
        return self._pressure_reduction

    @pressure_reduction.setter
    def pressure_reduction(self, value: float) -> None:
        if self._pressure_reduction > 0.0:
            self._pressure_reduction = self.pressure_reduction * value
        else:
            self._pressure_reduction = value

    @property
    def uncertainty(self) -> float:
        return self._uncertainty
    
    @uncertainty.setter
    def uncertainty(self, value: float) -> None:
        self._uncertainty = value


class State: 
    """
    Describes a state
    
    Properties:
        name (str): name of human activity
        id (int): identification number of activity
    """
    def __init__(self, name: str, id: int):
        self.name = name
        self._id = id
    
    def __repr__(self) -> str:
        return f"State({self.id}, {self.name})"

    @property
    def id(self) -> int:
        return self._id


class ActivityPressure:
    """
    Activity-Pressure pair

    Properties:
        name (str): description of activity-pressure pair
        id (int): identification number of activity-pressure pair (activity.id + pressure.id)
        expected (float): effect of the measure
        uncertainty (float): uncertainty of the measure's effect
        activity (Activity): activity object with id
        pressure (Pressure): pressure object with id

    """
    def __init__(self, activity: Type[Activity], pressure: Type[Pressure]) -> None:
        self.activity = activity
        self.pressure = pressure
        self.name = f"Activity {self.activity.id} and Pressure {self.pressure.id}"
        self._id = (activity.id, pressure.id)
        self._expected = 0.0
        self._min_expected = 0.0
        self._max_expected = 0.0
        self._uncertainty = 0.0

    def __repr__(self) -> str:
        return f"Activity-Pressure({self.id}, {self.name})"

    @property
    def id(self) -> int:
        return self._id

    @property
    def expected(self) -> np.ndarray:
        return self._expected
    
    @expected.setter
    def expected(self, value: Union[list, float, npt.ArrayLike]) -> None:
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, float):
            value = np.array([value])
        if value.max() >= 1.0 and value.min() <= 0.0:  
            raise ValueError("Value of expected value has to be between 0 and 1")
        self._expected = value
    
    @property
    def min_expected(self) -> np.ndarray:
        return self._min_expected
    
    @property
    def max_expected(self) -> np.ndarray:
        return self._max_expected
    
    @min_expected.setter
    def min_expected(self, value: Union[list, float, npt.ArrayLike]) -> None:
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, float):
            value = np.array([value])
        if value.max() >= 1.0 and value.min() <= 0.0:  
            raise ValueError("Value of mode has to be between 0 and 1")
    
    @max_expected.setter
    def max_expected(self, value: Union[list, float, npt.ArrayLike]) -> None:
        if isinstance(value, list):
            value = np.array(value)
        elif isinstance(value, float):
            value = np.array([value])
        if value.max() >= 1.0 and value.min() <= 0.0:  
            raise ValueError("Value of mode has to be between 0 and 1")


class Measure:
    """
    Measures are protection actions having effect on Activity-Pressure pairs.

    Properties:
        name (str): description of measure
        id (int): identification number of measure
        expected (float): effect of the measure, value between 0 and 1.0
        uncertainty (float): uncertainty of the measure's effect
        multiplier (float): effectivness of the measure, default 1.0
    """
    def __init__(self, name: str, id: int) -> None:
        self.name = name
        self._id = id
        self._activity_pressure = None
        self._states = None
        self._expected = None
        self._uncertainty = None
        self._effect = None
        self._multiplier = 1.0
        self._expected_distribution = None

    def __repr__(self) -> str:
        return  f"Measure({self.id}, {self.name})"

    def __hash__(self) -> int:
        if self.activity_pressure:
            return hash((self.id, self.activity_pressure.id))
        else:
            return hash((self.id, 0))
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)): return NotImplemented
        if self.activity_pressure:
            return self.id == self.id and self.activity_pressure.id == self.activity_pressure.id
        else:
            return self.id == self.id

    @property
    def id(self) -> int:
        return self._id

    @property
    def activity_pressure(self) -> type[ActivityPressure]:
        return self._activity_pressure

    @activity_pressure.setter
    def activity_pressure(self, instance: type[ActivityPressure]) -> None:
        if isinstance(instance, ActivityPressure):
            self._activity_pressure = instance
    
    @property
    def states(self) -> list[Type[State]]:
        return self._states

    def add_state(self, state: State) -> None:
        if self._states == None:
            self._states = []
        self._states.append(state)

    #
    # expected value's probability distribution, 
    # represented as a cumulative distribution, 
    # where every value represents the percentage chance (as fraction)
    # for a certain percentile range (boundaries dependant on length of distribution list)
    # that the expected value is lower or within that range
    #
    
    @property
    def expected_distribution(self) -> np.ndarray:
        return self._expected_distribution
    
    @expected_distribution.setter
    def expected_distribution(self, value: np.ndarray | list) -> None:
        if np.min(value) < 0 or np.max(value) > 1:
            raise ValueError("Value of expected value has to be between 0.0 and 1.0")
        self._expected_distribution = value
        self.expected = self.get_pick()

    #
    # expected value, represents the measure's expected activity-pressure reduction
    #

    @property
    def expected(self) -> float:
        return self._expected

    @expected.setter
    def expected(self, value: float) -> None:
        if value >= 1.0 and value <= 0.0:
            raise ValueError("Value of expected value has to be between 0.0 and 1.0")
        self._expected = value
        self._effect = self._multiplier * self._expected

    #
    # uncertainty
    #

    @property
    def uncertainty(self) -> float:
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value: float) -> None:
        if value >= 1.0 and value <= 0.0:
           raise ValueError("Value of uncertainty has to be between 0.0 and 1.0")
        self._uncertainty = value

    #
    # multiplier, affects the expected value
    #

    @property
    def multiplier(self) -> float:
        return self._multiplier
    
    @multiplier.setter
    def multiplier(self, value: float):
        if value < 0.0:
            raise ValueError("Value of multiplier has to be 0.0 or above!")
        self._multiplier = value
        self._effect = self._multiplier * self._expected

    #
    # effect, represents the measure's actual reduction
    #

    @property
    def effect(self):
        if self._effect == None:
            self._effect = self._expected * self._multiplier
        return self._effect
    
    #
    # utility functions
    #

    def get_probability_range(self, x1: int, x2: int = None):
        '''
        Returns probability of range [x1, x2], where x1 and x2 are percentages. 
        If only x1 is given, probability range [0, x1] is returned. 
        '''
        if self._expected_distribution is not None:
            if x1 < 0 or x2 > 100:
                raise ValueError("Given value(s) have to be within [0, 1]!")
            if x2 is not None:
                if x1 > x2:
                    raise ValueError("Lower range boundary has to be lower than higher boundary!")
                return self._expected_distribution[x2] - self._expected_distribution[x1]
            else:
                return self._expected_distribution[x1]
        else:
            raise ValueError("Expected value probability distribution not set!")
    
    def get_pick(self) -> float:
        '''
        Returns a random pick weighted by defined expected probability distribution
        '''
        if self._expected_distribution is not None:
            weights = np.zeros(self._expected_distribution.shape)
            for i in range(1, weights.size):
                weights[i] = self._expected_distribution[i] - self._expected_distribution[i-1]
            pick = np.random.random() * np.sum(weights)
            for k, val in enumerate(weights):
                if pick < val:
                    break
                pick -= val
            return pick
        else:
            raise ValueError("Expected value probability distribution not set!")

    def get_picks(self, size: int = None) -> list:
        '''
        Returns given number of random picks weighted by defined expected probability distribution.
        '''
        if self._expected_distribution is not None:
            if size is None:
                size = 1000
            picks = []
            weights = np.zeros(self._expected_distribution.shape)
            for i in range(1, weights.size):
                weights[i] = self._expected_distribution[i] - self._expected_distribution[i-1]
            for i in range(size):
                pick = np.random.random() * np.sum(weights)
                for k, val in enumerate(weights):
                    if pick < val:
                        break
                    pick -= val
                picks.append(pick)
            return picks
        else:
            raise ValueError("Expected value probability distribution not set!")


class Case:
    """
    Case keeps book on the most effective measure (with activity-pressure pair) in each CountryBasin.

    Properties:
        id (int): identification number of case
        most_effective (Measure): the most effective Measure in measures
        measures (list[Measure]): list of measures 
    """

    def __init__(self, id: int) -> None:
        self._id = id
        self._most_effective = None
        self._measures = []
    
    def __repr__(self) -> str:
        return f"Case {self.id}"

    @property
    def id(self) -> int:
        return self._id

    @property
    def most_effective(self) -> Type[Measure]:
        return self._most_effective
        
    @property
    def measures(self) -> list[Type[Measure]]:
        return self._measures
    
    @measures.setter
    def measures(self, measure: Type[Measure]) -> None:

        if self._most_effective == None:
            self._most_effective = measure

        elif self._most_effective.effect < measure.effect:
            self._most_effective = measure
        
        self._measures.append(measure)


class CountryBasin:
    """
    Describes areas in Baltic Sea region
    
    Properties:
        name (str): name of Baltic Sea region (country-basin pair)
        id (int): identification number of country-basin pair country id * 1000 + basin id
        basin_fraction (float): fraction of countries economic  in basin area
    """

    def __init__(self, id: int, name: str) -> None:
        self._id = id
        self._name = name
        self._basin_fraction = None

        self._measures = None
        self._cases = None
    
    def __repr__(self) -> str:
        return self._name

    @property
    def id(self) -> int:

        return self._id

    @property
    def coutry_id(self) -> int:

        return int(self._id / 1000)
    
    @property
    def basin_id(self) -> int:

        country_id = int(self._id / 1000)
        basin_id = self._id - country_id

        return basin_id

    @property
    def basin_fraction(self) -> float:

        return self._basin_fraction
    
    @basin_fraction.setter
    def basin_fraction(self, value):

        if value <= 0 or value > 1.0:
            raise ValueError("Fraction of basin area have to be between 0 and 1")

        self._basin_fraction = value

    @property
    def measures(self) -> list[Type[Measure]]:

        return list(self._measures)

    @measures.setter
    def measures(self, instance: Type[Measure]) -> None:

        if self._measures == None:
            self._measures = set()

        self._measures.add(instance)
    
    @property
    def cases(self) -> list[Type[Case]]:

        return self._cases
    
    @cases.setter
    def cases(self, instance: Type[Case]) -> None:

        if self._cases == None:
            self._cases = []
        
        self._cases.append(instance)

#EOF

