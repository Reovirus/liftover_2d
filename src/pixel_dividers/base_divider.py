from abc import ABC, abstractmethod
from typing import Any
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

# result_bin can duplicate if contains many source bins
_DIVIDER_SCHEMA = {
    'result_bin': TYPE_POLARS_INTEGER,
    'source_bin': TYPE_POLARS_INTEGER,
    'source_bin_part': TYPE_POLARS_INTEGER,
    'source_bin_count': TYPE_POLARS_INTEGER
}

class BaseDivider(ABC):
    """
    Base class for pixel dividers.
    This class should be inherited by all pixel divider implementations.
    """

    @abstractmethod
    def _calculate_divide_coeffs(self, *args, **Kwards) -> Any:
        pass

    @abstractmethod
    def _calculate_divide_matrix(self, *args, **Kwards) -> Any:
        pass

    @abstractmethod
    def _calculate_divide_matrix(self, *args, **Kwards) -> Any:
        pass

    @abstractmethod
    def divide_cells(self, 
        cooler_to_divide: CoolerDataReader,
        remap_matrix: pl.DataFrame
    ):
        pass