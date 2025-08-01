from abc import ABC, abstractmethod
from typing import Any, Optional, Literal
from src._utils import (
    TYPE_POLARS_INTEGER,
    TYPE_POLARS_SIGNED_INTEGER,
    TYPE_POLARS_STRING,
    TYPE_POLARS_FLOAT,
    TYPE_POLARS_BOOL,
    TYPE_BIN_ZONE,
    POLARS_VMAX
)

from src.readers import CoolerPolars
import numpy as np 

import polars as pl

def multinomial_norm(count, probs):
    probs = np.array(probs, dtype=float)
    sum_probs = probs.sum()
    extended_probs =  probs / sum_probs
    result = np.random.multinomial(count, extended_probs)
    return result.tolist()

def multinomial_with_remainder(count, probs):
    probs = np.array(probs, dtype=float)
    sum_probs = probs.sum()
    remainder = 1.0 - sum_probs
    extended_probs = np.append(probs, remainder)
    result = np.random.multinomial(count, extended_probs)
    return result[:-1].tolist()

class BaseDivider(ABC):
    """
    Base class for pixel dividers.
    This class should be inherited by all pixel divider implementations.
    """

    METRIC_NAME='BASE'
    NEED_NORMALISATION = True
    NEED_SAMPLING=True


    def __normalize_counts(self, contact_matrix, need_normalisation: bool=True):
        if not self.NEED_NORMALISATION and not self.NEED_SAMPLING:
            return contact_matrix.with_columns(
                pl.col(f"{self.METRIC_NAME}_weight").alias(f"{self.METRIC_NAME}_counts")
            )
        contact_matrix = contact_matrix.sort('source_bin_hash')
        grouped = contact_matrix.group_by("source_bin_hash").agg([
            pl.col(f"{self.METRIC_NAME}_weight").alias("weight_list"),
            pl.col("source_counts").first()
        ])
        if self.NEED_SAMPLING:
            if self.NEED_NORMALISATION:
                fun = multinomial_norm
            else: 
                fun = multinomial_with_remainder
            if self._mode == 'resample':

                grouped = grouped.with_columns(
                    pl.struct([f"{self.METRIC_NAME}_counts", "weight_list"]).map_elements(
                        lambda row: fun(row[f"{self.METRIC_NAME}_counts"], row["weight_list"])
                    ).alias("multinomial_samples")
                )
                contact_matrix.with_columns(
                    grouped.get_column('multinomial_samples').explode().alias(f"{self.METRIC_NAME}_counts")
                )
     
            elif self._mode == 'proportional':
                contact_matrix.with_columns(
                    (
                        pl.col(self._counts_column) * pl.col(f"{self.METRIC_NAME}_weight")
                    ).alias(f"{self.METRIC_NAME}_counts")
                )
            else:
                raise ValueError(f"Unacceptable mode {self._mode}. Possible modes are 'resample' and 'proportional' only.")
        else: 
            if self.NEED_NORMALISATION:
                grouped_contacts = grouped_contacts.with_columns(
                    pl.col("weight_list").list.sum().alias("weights_sum")
                ).explode(
                    "weight_list"
                ).with_columns(
                    (pl.col("weight_list") / pl.col("weights_sum")).alias("weight_list")
                )
                contact_matrix.with_columns(
                    grouped_contacts.get_column('weight_list').explode().alias(f"{self.METRIC_NAME}_weight")
                )
        return contact_matrix


    def __init__(self, mode: Literal['resample', 'proportional'], recalc_change_only:bool = False):
        self._mode = mode
        self._counts_column = 'counts'


    def set_prep_joined(self, matrix, counts_column):
        self._joined_bins=matrix
        self._counts_column=counts_column

    def join_matricies(self, source: CoolerPolars, remap_schema: pl.DataFrame):
        vmax = max(source.counts['bin1_id'].max(), source.counts['bin2_id'].max())
        if vmax > POLARS_VMAX**0.5 - 2:
            raise ValueError('Maximal bin id is too big')
        self._joined_bins = source.counts.select(
            pl.col('bin1_id').alias('source_bin_id_1'),
            pl.col('bin2_id').alias('source_bin_id_2'),
            pl.col('count').alias('source_counts'),
            (
                pl.col('bin1_id').cast(TYPE_POLARS_INTEGER) + vmax*pl.col('bin2_id').cast(TYPE_POLARS_INTEGER) 
            ).alias('source_bin_hash')
        ).join(
            remap_schema,
            right_on='source_bin',
            left_on='source_bin_id_1',
            how='inner'
        ).join(
            remap_schema,
            right_on='source_bin',
            left_on='source_bin_id_2',
            how='inner',
            suffix='_for_second_bin'
        ).rename(
            {
                'source_bin_chrom': 'source_bin_1_chrom',
                'source_bin_location': 'source_bin_1_location',
                'source_bin_region': 'source_bin_1_region',
                'target_bin': 'target_bin_1',
                'target_bin_chrom': 'target_bin_1_chrom',
                'target_bin_location': 'target_bin_1_location',
                'target_bin_region': 'target_bin_1_region',
                'source_bin_chrom_for_second_bin': 'source_bin_2_chrom',
                'source_bin_location_for_second_bin': 'source_bin_2_location',
                'source_bin_region_for_second_bin': 'source_bin_2_region',
                'target_bin_for_second_bin': 'target_bin_2',
                'target_bin_chrom_for_second_bin': 'target_bin_2_chrom',
                'target_bin_location_for_second_bin': 'target_bin_2_location',
                'target_bin_region_for_second_bin': 'target_bin_2_region',
            }
        )

    def predict(self, source, remap_schema: pl.DataFrame) -> pl.DataFrame:
        self.join_matricies(source, remap_schema)
        weight_matrix = self._calc_coeffs(self._joined_bins)
        weight_matrix = self.__normalize_counts(weight_matrix)
        return weight_matrix
    
    @abstractmethod
    def _compute_weights(self):
        pass
