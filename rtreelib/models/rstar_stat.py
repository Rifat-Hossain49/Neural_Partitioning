from typing import List, Dict, Union
from .entry_distribution import EntryDistribution
from ..rtree import EntryDivision

# Axis is now an integer (dimension index) or str for backward compat
Axis = Union[int, str]
Dimension = str  # 'min' or 'max'


class RStarStat:
    """
    Class used for caching metrics as part of the R*-Tree split algorithm. 
    Supports N-dimensional data where axes are indexed 0 to N-1.
    """

    def __init__(self, num_dims: int = 2):
        self.num_dims = num_dims
        # stat[axis_idx][dimension] = list of distributions
        self.stat: Dict[int, Dict[Dimension, List[EntryDistribution]]] = {}
        for axis in range(num_dims):
            self.stat[axis] = {'min': [], 'max': []}
        self.unique_distributions: List[EntryDistribution] = []

    def add_distribution(self, axis: int, dimension: Dimension, division: EntryDivision):
        """
        Adds a distribution of entries for the given axis and dimension.
        :param axis: Axis index (0 to num_dims-1)
        :param dimension: Dimension ('min' or 'max')
        :param division: Entry division
        """
        distribution = next((d for d in self.unique_distributions if d.is_division_equivalent(division)), None)
        if distribution is None:
            distribution = EntryDistribution(division)
            self.unique_distributions.append(distribution)
        
        if axis not in self.stat:
            self.stat[axis] = {'min': [], 'max': []}
        self.stat[axis][dimension].append(distribution)

    def get_axis_perimeter(self, axis: int) -> float:
        """
        Returns the total overall perimeter of all distributions along the given axis.
        :param axis: Axis index
        :return: Total overall perimeter for all distributions along the axis
        """
        distributions_min = self.stat.get(axis, {}).get('min', [])
        distributions_max = self.stat.get(axis, {}).get('max', [])
        return sum([d.perimeter for d in (distributions_min + distributions_max)])

    def get_axis_unique_distributions(self, axis: int) -> List[EntryDistribution]:
        """
        Returns a list of all unique entry distributions for a given axis
        :param axis: Axis index
        :return: List of unique entry distributions for the given axis
        """
        distributions = self.stat.get(axis, {}).get('min', []) + self.stat.get(axis, {}).get('max', [])
        return list(dict.fromkeys(distributions).keys())
