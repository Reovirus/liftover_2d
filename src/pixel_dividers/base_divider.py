from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, List, Tuple, Iterator
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
import itertools as it
import pandas as pd

from scipy.sparse import csr_matrix

def multinomial_norm(count, probs):
    probs = np.array(probs, dtype=float)
    sum_probs = probs.sum()
    extended_probs =  probs / sum_probs
    result = np.random.multinomial(count, extended_probs)
    return result.tolist()

def multinomial_with_remainder(count, probs):
    probs = np.array(probs, dtype=float)
    sum_probs = probs.sum()
    if sum_probs > 1.0:
        probs = probs / sum_probs
        remainder = 0.0
    else:
        remainder = 1.0 - sum_probs
    extended_probs = np.append(probs, remainder)
    result = np.random.multinomial(count, extended_probs)
    return result[:-1].tolist()

def _reflect_window_keep_diag(csr_window):
    diag = csr_matrix(csr_window.diagonal().reshape(-1, 1))
    to_add = csr_window.T
    to_add.setdiag(0)
    return csr_window + to_add

class BaseDivider(ABC):
    """
    Base class for pixel dividers.
    This class should be inherited by all pixel divider implementations.
    """

    METRIC_NAME='BASE'
    NEED_NORMALISATION=True
    NEED_SAMPLING=True
    COMPUTE_COUNTS=False


    def __normalize_counts(self, contact_matrix):
        contact_matrix = contact_matrix.filter(
            (pl.col(f"{self.METRIC_NAME}_weight").is_not_null()) 
            & (pl.col(f"{self.METRIC_NAME}_weight").is_not_nan()) 
        ).sort(
            'source_bin_hash'
        ).with_columns(
            pl.col('source_counts').fill_nan(0).alias('source_counts')
        )
        if self.COMPUTE_COUNTS:
            return contact_matrix.with_columns(
                pl.col(f"{self.METRIC_NAME}_weight").alias(f"{self.METRIC_NAME}_counts")
            )
        if self.NEED_SAMPLING or self.NEED_NORMALISATION:
            grouped = contact_matrix.group_by("source_bin_hash", maintain_order=True).agg(
                [
                    pl.col(f"{self.METRIC_NAME}_weight").alias("weight_list"),
                    pl.col("source_counts").first()
                ]
            ) 
        if self.NEED_SAMPLING:
            if self._mode == 'resample':
                if self.NEED_NORMALISATION:
                    fun = multinomial_norm
                else: 
                    fun = multinomial_with_remainder
                grouped = grouped.with_columns(
                    pl.struct([f"{self._counts_column}", "weight_list"]).map_elements(
                        lambda row: fun(row[f"{self._counts_column}"], row["weight_list"]),
                        return_dtype=pl.List(pl.Int64),
                        skip_nulls=False
                    ).alias("multinomial_samples")
                )
                contact_matrix = contact_matrix.with_columns(
                    grouped.get_column('multinomial_samples').explode().alias(f"{self.METRIC_NAME}_counts")
                )
            elif self._mode == 'proportional':
                if self.NEED_NORMALISATION:
                    cte = grouped.with_columns(
                        pl.col('weight_list').list.sum().alias('wieghs_sum')
                    ).explode(
                        'weight_list'
                    ).with_columns(
                        (pl.col('weight_list')/pl.col('wieghs_sum')).alias('weight_list')
                    ).get_column('weight_list')
                else:
                    cte = contact_matrix.get_column(f"{self.METRIC_NAME}_weight")

                contact_matrix = contact_matrix.with_columns(
                    (
                        pl.col('source_counts') * cte
                    ).alias(f"{self.METRIC_NAME}_counts")
                )
            else:
                raise ValueError(f"Unacceptable mode {self._mode}. Possible modes are 'resample' and 'proportional' only.")
        else: 
            if self.NEED_NORMALISATION:
                grouped_contacts = grouped_contacts.with_columns(
                    pl.col("weight_list_correct").list.sum().alias("weights_sum")
                ).explode(
                    "weight_list_correct"
                ).with_columns(
                    (pl.col("weight_list_correct") / pl.col("weights_sum")).alias("weight_list_corrected")
                )      
                contact_matrix.with_columns(
                    grouped_contacts.get_column('weight_list_corrected').alias(f"{self.METRIC_NAME}_weight")
                )
        return contact_matrix


    def __init__(self, mode: Literal['resample', 'proportional']='resample'):
        self._mode = mode
        self._counts_column = 'source_counts'
        self._return_weights_only = False
        self._not_normalize = False

    #method for pipline initialization
    def set_prep_joined(self, matrix, counts_column, not_normalize):
        self._joined_bins=matrix
        self._counts_column=counts_column
        self._not_normalize = not_normalize

    def join_matricies(self, source: CoolerPolars, remap_schema: pl.DataFrame):
        vmax = int(source.n_bins)
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

    def _predict_in_pipeline(self, **args):
        weight_matrix = self._compute_weights(**args)
        if not self._not_normalize:
            weight_matrix = self.__normalize_counts(weight_matrix)
        return weight_matrix
    
    @abstractmethod
    def _compute_weights(self, source_hic, **args):
        pass

    def predict(self, source, remap_schema: pl.DataFrame, return_format: Literal['full', 'compact']='compact', **args) -> pl.DataFrame:
        self.join_matricies(source, remap_schema)
        weight_matrix = self._compute_weights(source)
        weight_matrix = self.__normalize_counts(weight_matrix)
        self.weight_matrix = weight_matrix
        if return_format == 'full':
            return weight_matrix
        elif return_format == 'compact':
            return weight_matrix.select(
                pl.col('target_bin_1').alias('bin1_id'),
                pl.col('target_bin_2').alias('bin2_id'),
                pl.col(f'{self.METRIC_NAME}_counts').alias('count')
            ).with_columns(
                [
                    pl.when(
                        pl.col("bin1_id") < pl.col("bin2_id")
                    ).then(
                        pl.struct(
                            [
                                pl.col("bin1_id").alias('bin_1_corr'),
                                pl.col("bin2_id").alias('bin_2_corr')
                            ]
                        )
                    ).otherwise(
                        pl.struct(
                            [
                                pl.col("bin2_id").alias('bin_1_corr'),
                                pl.col("bin1_id").alias('bin_2_corr')
                            ]
                        )
                    ).struct.unnest()
                ]
            ).group_by(
                ["bin_1_corr", "bin_2_corr"], maintain_order=True
            ).agg(
                pl.sum("count").alias("count")
            ).select(
                pl.col("bin_1_corr").alias('bin1_id'),
                pl.col("bin_2_corr").alias('bin2_id'),
                pl.col("count")
            )
        else:
            raise ValueError(f"Unacceptable return format {return_format}. Possible formats are 'full' and 'compact'.")
        

class BaseNeigborUsingDivider(BaseDivider):
    METRIC_NAME='BASE_NEIGBUR'
    NEED_NORMALISATION=False
    NEED_SAMPLING=True
    COMPUTE_COUNTS=False

    @staticmethod
    def __calculate_small_window(bin1, bin2, sparse_contacts, window_size, bbox):
        #fix fckng problems with uint64 overflow
        loc_hor = int(bin1 - bbox[0])
        #same. kak ze ya ustal ot vsego
        loc_ver = int(bin2 - bbox[2])

        start_i = max(0, loc_hor - window_size)
        end_i   = min(sparse_contacts.shape[0], loc_hor + window_size + 1)
        start_j = max(0, loc_ver - window_size)
        end_j   = min(sparse_contacts.shape[1], loc_ver + window_size + 1)
        
        window = sparse_contacts[start_i:end_i, start_j:end_j]
        location_in_window = (loc_hor - start_i, loc_ver - start_j)
        
        return window, location_in_window

    def __init__(self, ident: int=4, scale_factor: int=5, mode: Literal['resample', 'proportional']='resample'):
        self._ident = ident
        self._scale_factor = scale_factor
        super().__init__(mode=mode)

    @abstractmethod
    def _process_one_window(self, counts_arr, location):
        pass

    def __calculate_one_pixel(self, sourse, region_in_rate_1: List[float], region_in_rate_2: List[float]) -> float:
        minimal_divider = 1/self._scale_factor
        # (x + y/2)//y = math rounded locartion
        slice_bin_1_start = (region_in_rate_1[0] + 0.5*minimal_divider)//minimal_divider
        slice_bin_1_end = (region_in_rate_1[1] + 0.5*minimal_divider)//minimal_divider

        slice_bin_2_start = (region_in_rate_2[0] + 0.5*minimal_divider)//minimal_divider
        slice_bin_2_end = (region_in_rate_2[1] + 0.5*minimal_divider)//minimal_divider

        weight = sourse[int(slice_bin_1_start):int(slice_bin_1_end), int(slice_bin_2_start):int(slice_bin_2_end)].sum()
        return weight
    
    def __generate_bbox(self) -> Iterator[Tuple[Tuple[str, str], Tuple[int, int, int, int]]]:
        for chrom_1, chrom_2 in self._joined_bins.select(
            pl.col('source_bin_1_chrom'),
            pl.col('source_bin_2_chrom')
        ).unique().iter_rows():
            yield (
                self.__bins_locations.loc[chrom_1, 'min_bin'],
                self.__bins_locations.loc[chrom_1, 'max_bin'],
                self.__bins_locations.loc[chrom_2, 'min_bin'],
                self.__bins_locations.loc[chrom_2, 'max_bin']
            ), (chrom_1, chrom_2)

    def _compute_weights(self, source):
        self.__counts_per_pixel_tmp = source.counts.to_pandas()
        self.__counts_per_pixel_tmp.index = pd.MultiIndex.from_frame(
            self.__counts_per_pixel_tmp.loc[:, ['bin1_id', 'bin2_id']]
        )
        self.__counts_per_pixel_tmp.drop(columns=['bin1_id', 'bin2_id'], inplace = True)

        #chromosome_borders
        self.__bins_locations = source.bins.group_by(
            'chrom'
        ).agg(
            [
                pl.min('bin_id').alias('min_bin'),
                pl.max('bin_id').alias('max_bin')
            ]
        ).to_pandas().set_index('chrom')

        self._joined_bins = self._joined_bins.with_columns(
           (
                (pl.col('source_bin_1_region').list.get(1) - pl.col('source_bin_1_region').list.get(0))
                / (pl.col('source_bin_1_location').list.get(1) - pl.col('source_bin_1_location').list.get(0))
            ).alias('bin_1_linear_part'),
            
            (
                (pl.col('source_bin_2_region').list.get(1) - pl.col('source_bin_2_region').list.get(0))
                / (pl.col('source_bin_2_location').list.get(1) - pl.col('source_bin_2_location').list.get(0))
            ).alias('bin_2_linear_part'),
            
            pl.concat_list([
                (pl.col('source_bin_1_region').list.get(0) - pl.col('source_bin_1_location').list.get(0))
                    / 
                (pl.col('source_bin_1_location').list.get(1) - pl.col('source_bin_1_location').list.get(0)),
                (pl.col('source_bin_1_region').list.get(1) - pl.col('source_bin_1_location').list.get(0))
                    /  
                (pl.col('source_bin_1_location').list.get(1) - pl.col('source_bin_1_location').list.get(0))
            ]).alias("source_bin_1_region_rate"),
            
            pl.concat_list([
                (pl.col('source_bin_2_region').list.get(0) - pl.col('source_bin_2_location').list.get(0)) 
                / 
                (pl.col('source_bin_2_location').list.get(1) - pl.col('source_bin_2_location').list.get(0)),
                (pl.col('source_bin_2_region').list.get(1) - pl.col('source_bin_2_location').list.get(0)) 
                    / 
                (pl.col('source_bin_2_location').list.get(1) - pl.col('source_bin_2_location').list.get(0))
            ]).alias("source_bin_2_region_rate")
        ).filter(
            (pl.col('bin_1_linear_part') >= (1.0/self._scale_factor))
            & (pl.col('bin_2_linear_part') >= (1.0/self._scale_factor))
        ).with_row_count(
            name='row_number'
        )  

        weights = np.ones(self._joined_bins.height, dtype=np.float32)
        
        sparce_contacts = csr_matrix(
            (
                source.counts['count'].to_numpy(), 
                (source.counts['bin1_id'].to_numpy(), source.counts['bin2_id'].to_numpy())
            ), 
            shape=(source.n_bins, source.n_bins)
        )
        sparce_contacts = _reflect_window_keep_diag(sparce_contacts).toarray()

        chroms = (None, None)

        to_one_trashold = 1 - (1.0/self._scale_factor)

        for bbox, chroms in self.__generate_bbox():
            curr_subset = sparce_contacts[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
            wr = None
            bin_1_id = None
            bin_2_id = None
            for row in self._joined_bins.filter(
                (pl.col('source_bin_1_chrom') == chroms[0]) 
                & (pl.col('source_bin_2_chrom') == chroms[1])
                & (pl.col('bin_1_linear_part') <= to_one_trashold)
                & (pl.col('bin_2_linear_part') <= to_one_trashold)        
            ).select(
                pl.col('row_number'),
                pl.col('source_bin_id_1').alias('bin1_id'),
                pl.col('source_bin_id_2').alias('bin2_id'),
                pl.col('source_bin_1_region_rate'),
                pl.col('source_bin_2_region_rate'),
            ).iter_rows(named=True):
                if bin_1_id != row['bin1_id'] or bin_2_id != row['bin2_id'] or wr is not None:
                    window, location = self.__calculate_small_window(
                        row['bin1_id'], 
                        row['bin2_id'], 
                        curr_subset, 
                        self._ident, 
                        bbox
                    )

                    #empty window
                    if window.shape[0] == 0 or window.shape[1] == 0:
                        print('Empty window for row', row)
                        print('bbox', bbox)
                        weights[row['row_number']] = 0.0
                        continue
                    weights_raw = self._process_one_window(window, location)
                    wr = weights_raw
                    bin_1_id = row['bin1_id']
                    bin_2_id = row['bin2_id']
                else:
                    weights_raw = wr
                weights[row['row_number']] = self.__calculate_one_pixel(
                    weights_raw,
                    row['source_bin_1_region_rate'], 
                    row['source_bin_2_region_rate']
                )
        self._joined_bins = self._joined_bins.with_columns(
            pl.Series(weights).alias(f"{self.METRIC_NAME}_weight")
        )
        return self._joined_bins
