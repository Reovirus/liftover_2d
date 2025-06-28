import cooler
import polars as pl
import polars_bio as pb

from src.readers import (
    CoolerReader,
    ChainReader
)
from src._utils import (
    TYPE_POLARS_INTEGER,
    TYPE_POLARS_STRING,
    TYPE_BIN_ZONE
)

_BINS_MATCH_SCHEMA = {
    'source_bin': TYPE_POLARS_INTEGER,
    'source_bin_chrom': TYPE_POLARS_STRING,
    'source_bin_location': TYPE_BIN_ZONE,
    'source_bin_region': TYPE_BIN_ZONE,

    'target_bin': TYPE_POLARS_INTEGER,
    'target_bin_chrom': TYPE_POLARS_STRING,
    'target_bin_location': TYPE_BIN_ZONE,
    'target_bin_region': TYPE_BIN_ZONE
}

def remap_bins(
        source: CoolerReader, target: CoolerReader, chains: ChainReader
) -> pl.DataFrame:
    #WTF? So I just tried to run it, but this asshole gives some crap instead of columns 
    #1 – what’s wrong with cols2 and cols1, why the hell are they reversed
    # 2 – why the fuck is such bullshit coming out in the output names).
    # I've killed to days to explote cols1 and cols2 changind
    #version in reqs file pls - I hope it'll be fixerd soon
    joined_source = pb.overlap(
        df1=source.bins,
        df2=chains.blocks.select(
            pl.col("name_chain_source").alias("chrom_source"),
            pl.col("start_aln_source").alias("start_source"),
            pl.col("end_aln_source").alias("end_source"),
            pl.col("name_chain_target").alias("chrom_target"),
            pl.col("start_aln_target").alias("start_target"),
            pl.col("end_aln_target").alias("end_target")
        ),
        suffixes=('_bin_source', '_blocks'),
        cols2=('chrom', 'start', 'end'), 
        cols1=('chrom_source', 'start_source', 'end_source'),
        output_type='polars.DataFrame'
    ).with_columns(
        [
            pl.max_horizontal("start_source_blocks", "start_bin_source").alias("source_start"),
            pl.min_horizontal("end_source_blocks", "end_bin_source").alias("source_end"),
            (
                pl.col('end_target_bin_source') - (pl.col('end_source_blocks') - pl.min_horizontal("end_source_blocks", "end_bin_source"))
            ).alias('new_end_target'),
            (
                pl.col('start_target_bin_source') + (pl.min_horizontal("start_source_blocks", "start_bin_source") - pl.col('start_source_blocks'))
            ).alias('new_start_target')
        ]
    ).select(
        pl.col('bin_id_blocks').alias('source_bin_id'),
        pl.col('start_bin_source').alias('source_bin_start'),
        pl.col('end_bin_source').alias('source_bin_end'),
        pl.col('chrom_bin_source').alias('source_bin_chrom'),
        pl.col('source_start'),
        pl.col('source_end'),
        pl.col('new_end_target'),
        pl.col('new_start_target')
    )
    result = pb.overlap(
        df1=target.bins,
        df2=joined_source,
        suffixes=('_bin_target', ''),
        cols2=('chrom', 'start', 'end'), 
        cols1=('source_bin_chrom', 'new_start_target', 'new_end_target'),
        output_type='polars.DataFrame'
    ).with_columns(
        [
            pl.max_horizontal("new_start_target_bin_target", "start_bin_target").alias("target_start"),
            pl.min_horizontal("new_end_target_bin_target", "end_bin_target").alias("target_end"),
        ]
    ).select(
        pl.col('source_bin_id_bin_target').alias('source_bin'),
        pl.col("source_bin_chrom"),
        pl.concat_list(["source_bin_start_bin_target", "source_bin_end_bin_target"]).alias("source_bin_location"),
        pl.concat_list(["source_start_bin_target", "source_end_bin_target"]).alias("source_bin_region"),
        pl.col('bin_id').alias('target_bin'),
        pl.col("chrom_bin_target").alias("target_bin_chrom"),
        pl.concat_list(["start_bin_target", "end_bin_target"]).alias("target_bin_location"),
        pl.concat_list(["target_start", "target_start"]).alias("target_bin_region"),
    ).cast(
        _BINS_MATCH_SCHEMA
    )
    return result