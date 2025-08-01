import pandas as pd
import polars as pl
from cooler.core import (
    CSRReader,
    DirectRangeQuery2D
)
import cooler
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Union, Tuple, Generator

from src._utils import (
    TYPE_POLARS_INTEGER,
    TYPE_POLARS_STRING
)

_COOLER_CONTACTS_SCHEMA = {
    'bin1_id': TYPE_POLARS_INTEGER,
    'bin2_id': TYPE_POLARS_INTEGER,
    'count': TYPE_POLARS_INTEGER,
}
_COOLER_BINS_SCHEMA = {
    'bin_id': TYPE_POLARS_INTEGER,
    'chrom': TYPE_POLARS_STRING,
    'start': TYPE_POLARS_INTEGER,
    'end': TYPE_POLARS_INTEGER,
}

@dataclass(frozen=True)
class CoolerPolars:
    n_bins: int
    bins: pl.DataFrame
    counts: pl.DataFrame

    def to_cooler(
        self, 
        file_path:str,
        assembly=None, 
        symmetric_upper=True,
        return_cooler: bool = True
    ):
        bins_pandas = self.bins.to_pandas()
        bins_pandas.index = bins_pandas.bin_id
        bins_pandas.pop('bin_id')
        pixels_pandas = self.counts.to_pandas()
        cooler.create_cooler(
            cool_uri=file_path,
            bins=bins_pandas,
            pixels=pixels_pandas,
            assembly=assembly,
            symmetric_upper=symmetric_upper
        )
        if return_cooler:
            return cooler.Cooler(file_path)
        return
        
    def __calc_full_info(self):
        self.__full_info = self.__counts.join(
            self.__bins, left_on='bin1_id', right_on='bin_id', how='left'
        ).join(
            self.__bins, left_on='bin2_id', right_on='bin_id', how='left', suffix="_bin2"
        ).select(
            [
                pl.col('bin1_id'),
                pl.col('bin2_id'),
                pl.col('count'),
                pl.col('chrom').alias('chrom1'),
                pl.col('start').alias('start1'),
                pl.col('end').alias('end1'),
                pl.col('chrom_bin2').alias('chrom2'),
                pl.col('start_bin2').alias('start2'),
                pl.col('end_bin2').alias('end2')
            ]
        )

    @property
    def joined_bins(self):
        if self.__full_info is None:
            self.__calc_full_info()
        return self.__full_info


def __process_chunk(chunk):
    df = pd.DataFrame(chunk, columns=["bin1_id", "bin2_id", "count"])
    pl_df = pl.from_pandas(df).cast(_COOLER_CONTACTS_SCHEMA)
    return pl_df


def read_cooler(cooler_file: Union[cooler.Cooler, str], chunk_size: int=100_000, max_workers: int=4):
    if isinstance(cooler_file, str):
        cooler_readed = cooler.Cooler(cooler_file)
    else:
        cooler_readed = cooler_file
    n_bins = cooler_readed.info['nbins']
    bins = pl.from_pandas(cooler_readed.bins()[:].reset_index(names='bin_id').loc[:, ['bin_id', 'chrom', 'start', 'end']]).cast(_COOLER_BINS_SCHEMA)

    bbox = (0, n_bins, 0, n_bins)
    h5 = cooler_readed.open("r")
    reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
    field = "count"
    engine = DirectRangeQuery2D(reader=reader, field=field, bbox=bbox, chunksize=chunk_size)
    results = []
    with ThreadPoolExecutor(max_workers==max_workers) as executor:
        futures = [executor.submit(__process_chunk, chunk) for chunk in engine]
        for future in as_completed(futures):
            results.append(future.result())
    cnts = pl.concat(results).with_columns(
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

    return CoolerPolars(
        n_bins=n_bins,
        bins=bins,
        counts=cnts
    )


def read_cooler_chunk(cooler_file: Union[cooler.Cooler, str], chunk_size: int=100_000):
    if isinstance(cooler_file, str):
        cooler_readed = cooler.Cooler(cooler_file)
    else:
        cooler_readed = cooler_file
    n_bins = cooler_readed.info['nbins']
    bins = pl.from_pandas(cooler_readed.bins()[:].reset_index(names='bin_id').loc[:, ['bin_id', 'chrom', 'start', 'end']]).cast(_COOLER_BINS_SCHEMA)

    bbox = (0, n_bins, 0, n_bins)
    h5 = cooler_readed.open("r")
    reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
    field = "count"
    engine = DirectRangeQuery2D(reader=reader, field=field, bbox=bbox, chunksize=chunk_size)
    for chunk in engine: 
        res = __process_chunk(chunk)
        cnts = res.with_columns(
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

        yield CoolerPolars(
            n_bins=n_bins,
            bins=bins,
            counts=cnts
        )


def __generate_sliding_windows(
    n_bins: int,
    size: int,
    ovl: int
) -> Generator[Tuple[int, int, int, int], None, None]:
    """Yield (row_start, row_stop, col_start, col_stop) windows."""
    step = size - ovl
    for row_start in range(0, n_bins, step):
        row_stop = min(row_start + size, n_bins)
        for col_start in range(0, n_bins, step):
            col_stop = min(col_start + size, n_bins)
            yield (row_start, row_stop, col_start, col_stop)


def read_cooler_sqaure(cooler_file: Union[cooler.Cooler, str], square_size: int=700, square_overlap: int = 5, chunk_size=None):
    chunk_size = square_size**2 if chunk_size is None else chunk_size
    if isinstance(cooler_file, str):
        cooler_readed = cooler.Cooler(cooler_file)
    else:
        cooler_readed = cooler_file
    n_bins = cooler_readed.info['nbins']
    bins = pl.from_pandas(cooler_readed.bins()[:].reset_index(names='bin_id').loc[:, ['bin_id', 'chrom', 'start', 'end']]).cast(_COOLER_BINS_SCHEMA)

    h5 = cooler_readed.open("r")
    reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
    field = "count"

    for bbox in __generate_sliding_windows(
        n_bins=n_bins,
        size=square_size,
        ovl=square_overlap
    ):
        results = []
        engine = DirectRangeQuery2D(reader=reader, field=field, bbox=bbox, chunksize=chunk_size)
        for chunk in engine: 
            res = __process_chunk(chunk)
            results.append(res)
        
        
        cnts = pl.concat(results).with_columns(
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
        yield CoolerPolars(
            n_bins=n_bins,
            bins=bins,
            counts=cnts
        )