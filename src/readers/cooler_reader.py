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

class CoolerPolars:
    @staticmethod
    def process_chunk(chunk):
        df = pd.DataFrame(chunk, columns=["bin1_id", "bin2_id", "count"])
        pl_df = pl.from_pandas(df).cast(_COOLER_CONTACTS_SCHEMA)
        return pl_df

    def __init__(self, cooler, chunk_size=100_000, max_workers=4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
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

    @property
    def counts(self):
        return self.__counts

    @property
    def bins(self):
        return self.__bins

    @property
    def n_bins(self):
        return self.__n_bins
