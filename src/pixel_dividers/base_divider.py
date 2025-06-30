from abc import ABC, abstractmethod
from typing import Any, Optional, Literal
from numba import njint
from src._utils import (
    TYPE_POLARS_INTEGER,
    TYPE_POLARS_SIGNED_INTEGER,
    TYPE_POLARS_STRING,
    TYPE_POLARS_FLOAT,
    TYPE_POLARS_BOOL,
    TYPE_BIN_ZONE
)

from src.readers import CoolerDataReader

import polars as pl

class BaseDivider(ABC):
    """
    Base class for pixel dividers.
    This class should be inherited by all pixel divider implementations.
    """

    @njint
    def calc_resample(self, arr_arr):
        pass

    @staticmethod
    def get_hash():
        pass

    def __normalize_counts(self, contact_matrix, need_normalisation: bool=True):
        contact_matrix = contact_matrix.sort('parent_hash')
        grouped_contacts = contact_matrix.groupby("parent_hash").agg(
            pl.col("weight").list().alias("weight_list")
        )
        
        if need_normalisation:
            grouped_contacts.with_columns(
                (pl.col("weight_list").arr.eval(pl.element() / pl.col("weight_list").arr.sum())).alias("weight_list")
            )

        if self.__mode == 'resample':
            pass
        elif self.__mode == 'proportional':
            pass
        else:
            raise ValueError(f"Unacceptable mode {self.__mode}. Possible modes are 'resample' and 'proportional' only.")


    def __init__(self, mode: Literal['resample', 'proportional'], recalc_change_only:bool = True):
        self.__mode = mode


    @abstractmethod
    def _calculate_divide_coeffs(self, *args, **Kwards) -> Any:
        pass

    @abstractmethod
    def _calculate_counts(self, *args, **Kwards) -> Any:
        pass

    @abstractmethod
    def __calculate_weights(self, *args, **Kwards) -> Any:
        if self.__mode == 'resample':
            pass
        elif self.__mode == 'proportional':
            pass
        else:
            raise ValueError(f"Unacceptable mode {self.__mode}. Possible modes are 'resample' and 'proportional' only.")

    @abstractmethod
    def predict(self, 
        cooler_to_divide: CoolerDataReader,
        remap_matrix: pl.DataFrame,
        cooler_to_get: Optional[pl.DataFrame]=None,
    ):
        if not cooler_to_divide:
            pass

