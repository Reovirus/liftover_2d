import polars as pl
import numpy as np
from numba import njit, prange

TYPE_POLARS_INTEGER = pl.UInt64
TYPE_POLARS_SIGNED_INTEGER = pl.Int64
TYPE_POLARS_STRING  = pl.String
TYPE_POLARS_FLOAT   = pl.Float64
TYPE_BIN_ZONE   = pl.List(pl.Float64)
TYPE_POLARS_BOOL = pl.Boolean
POLARS_VMAX = pl.DataFrame({'vmax':0}).with_columns( vmax=(pl.UInt64.max()//2) ).to_numpy()[0, 0]


@njit
def multinomial_from_binomial(n, pvals):
    out = np.empty(pvals.shape[0], dtype=np.int64)
    remaining_n = n
    remaining_p = 1.0
    for i in range(pvals.shape[0] - 1):
        p = pvals[i] / remaining_p
        draw = np.random.binomial(remaining_n, p)
        out[i] = draw
        remaining_n -= draw
        remaining_p -= pvals[i]
    out[-1] = remaining_n
    return out

@njit
def simulate_multinomial(counts, locations_st, locations_en, weights):
    outs = np.zeros(weights.shape[0], dtype=np.int64)
    for hash_position in prange(counts.shape[0]):
        int_start = locations_st[hash_position]
        int_end = locations_en[hash_position]
        sim_vals = multinomial_from_binomial(n=counts[hash_position], pvals=weights[int_start:int_end])
        for out_position in prange(int_start, int_end):
            loc_in_sim = out_position - int_start
            outs[out_position] = sim_vals[loc_in_sim]
    return outs