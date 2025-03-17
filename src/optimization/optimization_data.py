############################################################################
### QPMwP - CLASS OptimizationData
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------


# Standard library imports
from typing import Optional

# Third party imports
import pandas as pd


class OptimizationData(dict):
    """
    A class to handle optimization data,
    allowing for alignment of dates and lagging of variables.

    Parameters:
    align (bool): Whether to align dates across all variables.
    lags (dict): Dictionary specifying the lag for each variable.
    kwargs: Additional keyword arguments to initialize the dictionary.
    """

    def __init__(self, align: bool = True, lags: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self
        if len(lags) > 0:
            for key in lags.keys():
                self[key] = self[key].shift(lags[key])
        if align:
            self.align_dates()

    def align_dates(self, variable_names: Optional[list[str]] = None) -> None:
        """
        Align dates across specified variables.

        Parameters:
        variable_names (Optional[list[str]]): List of variable names to align.
        If None, align all variables.
        """
        if variable_names is None:
            variable_names = self.keys()
        index = self.intersecting_dates(variable_names=list(variable_names))
        for key in variable_names:
            self[key] = self[key].loc[index]
        return None

    def intersecting_dates(
        self, variable_names: Optional[list[str]] = None, dropna: bool = True
    ) -> pd.DatetimeIndex:
        """
        Find intersecting dates across specified variables.

        Parameters:
        variable_names (Optional[list[str]]): List of variable names to consider.
        If None, consider all variables.
        dropna (bool): Whether to drop NaN values.

        Returns:
        pd.DatetimeIndex: Intersecting dates.
        """
        if variable_names is None:
            variable_names = list(self.keys())
        if dropna:
            for variable_name in variable_names:
                self[variable_name] = self[variable_name].dropna()
        index = self.get(variable_names[0]).index
        for variable_name in variable_names:
            index = index.intersection(self.get(variable_name).index)
        return index
