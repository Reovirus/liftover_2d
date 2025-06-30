import polars as pl

TYPE_POLARS_INTEGER = pl.UInt64
TYPE_POLARS_SIGNED_INTEGER = pl.Int64
TYPE_POLARS_STRING  = pl.String
TYPE_POLARS_FLOAT   = pl.Float64
TYPE_BIN_ZONE   = pl.List(pl.Float64)
TYPE_POLARS_BOOL = pl.Boolean
POLARS_VMAX = pl.DataFrame({'vmax':0}).with_columns( vmax=(pl.UInt64.max()//2) ).to_numpy()[0, 0]