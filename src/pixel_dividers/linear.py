import polars as pl

from src.pixel_dividers.base_divider import BaseDivider
from src._utils import TYPE_POLARS_SIGNED_INTEGER


class LinearDivider(BaseDivider):
    METRIC_NAME='LINEAR_1D'
    NEED_NORMALISATION=False
    NEED_SAMPLING=True
    COMPUTE_COUNTS=False

    def __init__(self, mode='resample'):
        super().__init__(mode=mode)

    def _compute_weights(self, source_hic):
        self._joined_bins = self._joined_bins.with_columns(
            (
                (
                    pl.col('source_bin_1_region').list.get(1) - pl.col('source_bin_1_region').list.get(0)
                )/(
                    pl.col('source_bin_1_location').list.get(1) - pl.col('source_bin_1_location').list.get(0)
                )
            ).alias('bin_1_linear_part'),
            (
                (
                     pl.col('source_bin_2_region').list.get(1) - pl.col('source_bin_2_region').list.get(0)
                )/(
                    pl.col('source_bin_2_location').list.get(1) - pl.col('source_bin_2_location').list.get(0)
                )
            ).alias('bin_2_linear_part')
        ).with_columns(
            (pl.col('bin_1_linear_part')*pl.col('bin_2_linear_part') ).alias(f"{self.METRIC_NAME}_weight")
        )

        return self._joined_bins
