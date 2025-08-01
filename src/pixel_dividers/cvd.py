import polars as pl

from .base_divider import BaseDivider
from src._utils import TYPE_POLARS_SIGNED_INTEGER


class CVDNorm(BaseDivider):
    METRIC_NAME='CVD'
    NEED_NORMALISATION=True
    NEED_SAMPLING=True

    def __init__(self, cis, trans, mode):
        self.__cis = dict(cis)
        self.__trans = trans
        super().__init__(mode=mode)

    def _compute_weights(self):
        self._joined_bins = self._joined_bins.with_columns(
            (
                pl.col('target_bin_1').cast(TYPE_POLARS_SIGNED_INTEGER) - pl.col('target_bin_2').cast(TYPE_POLARS_SIGNED_INTEGER)
            ).abs().alias('bin_distance')
        ).with_columns(
            pl.when(
                pl.col("target_bin_1_chrom") == pl.col("target_bin_2_chrom")
            ).then( 
                pl.col('bin_distance').replace_strict(self.__cis, default=self.__trans) 
            ).otherwise(
                self.__trans
            ).alias(f"{self.METRIC_NAME}_weight")
        )