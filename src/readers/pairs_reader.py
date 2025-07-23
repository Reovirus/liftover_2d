import polars as pl

from pairtools.lib.pairsio import read_pairs

from src._utils import TYPE_POLARS_INTEGER, TYPE_POLARS_STRING, TYPE_POLARS_SIGNED_INTEGER

_PAIRS_READ_SCHEMA = {
    'readid': TYPE_POLARS_STRING,
    'chrom1': TYPE_POLARS_STRING,
    'pos1': TYPE_POLARS_INTEGER,
    'chrom2': TYPE_POLARS_STRING,
    'pos2': TYPE_POLARS_INTEGER,
    'strand1': TYPE_POLARS_STRING,
    'strand2': TYPE_POLARS_STRING,
    'pair_type': TYPE_POLARS_STRING,
    'mapq1': TYPE_POLARS_INTEGER,
    'mapq2': TYPE_POLARS_INTEGER
}

def read_pairs(path):
    pairs_df, header, chromsizes = read_pairs(path)
    # readID cannot be correctly used in overlap
    pairs_polars = pl.from_pandas(pairs_df).rename({'readID': 'readid'}).cast(_PAIRS_READ_SCHEMA)
    return pairs_polars 