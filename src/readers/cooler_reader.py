import pandas as pd
import polars as pl
from cooler.core import (
    CSRReader,
    DirectRangeQuery2D
)
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class CoolerReader:
    @staticmethod
    def process_chunk(chunk):
        df = pd.DataFrame(chunk, columns=["bin1_id", "bin2_id", "count"])
        pl_df = pl.from_pandas(df).cast(_COOLER_CONTACTS_SCHEMA)
        return pl_df

    def __init__(self, cooler, chunk_size=100_000, max_workers=4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.__full_info = None  
        self.__read_cooler(cooler)

    def __read_cooler(self, cooler):
        self.__n_bins = cooler.info['nbins']
        self.__bins = pl.from_pandas(cooler.bins()[:].reset_index(names='bin_id').loc[:, ['bin_id', 'chrom', 'start', 'end']]).cast(_COOLER_BINS_SCHEMA)

        bbox = (0, self.__n_bins, 0, self.__n_bins)
        h5 = cooler.open("r")
        reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
        field = "count"

        engine = DirectRangeQuery2D(reader, field, bbox, self.chunk_size)

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in engine]
            for future in as_completed(futures):
                results.append(future.result())

        self.__counts = pl.concat(results)

    def calc_full_info(self):
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
    def counts(self):
        return self.__counts

    @property
    def bins(self):
        return self.__bins

    @property
    def n_bins(self):
        return self.__n_bins
    
    @property
    def joined_bins(self):
        if self.__full_info is None:
            self.calc_full_info()
        return self.__full_info

