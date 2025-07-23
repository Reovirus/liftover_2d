import polars as pl
import polars_bio as pb

from src.readers.chain_reader import ChainReader

def remap_pairs(pairs_df: pl.DataFrame, chains: ChainReader, old_border_shift: int=150, new_border_shift: int=150):
    overlapped_source_pairs = pb.overlap(
        df1=pairs_df.with_columns(
            pl.col('pos1').alias('pos_1_end')
        ),
        df2=chains.blocks.select(
                pl.col("name_chain_source").alias("chrom_source"),
                pl.col("start_aln_source").alias("start_source"),
                pl.col("end_aln_source").alias("end_source"),
                pl.col("name_chain_target").alias("chrom_target"),
                pl.col("start_aln_target").alias("start_target"),
                pl.col("end_aln_target").alias("end_target"),
                pl.col("strand_chain_target"),
                pl.col("strand_chain_source")
            ),
        suffixes=('_pairs', '_blocks'),
        cols2=('chrom1', 'pos1', 'pos_1_end'),
        cols1=('chrom_source', 'start_source', 'end_source'),
        output_type='polars.DataFrame'
    ).with_columns(
        pl.when(
            pl.col('strand_chain_target_pairs') == '+'
        ).then(
            pl.col('end_target_pairs') - (pl.col('end_source_blocks') - pl.col('pos1_pairs'))
        ).otherwise(
            pl.col('start_target_pairs') + (pl.col('end_source_blocks') - pl.col('pos1_pairs'))
        ).alias('pos_1_relocated'),

        pl.when(
            pl.col('strand_chain_target_pairs') == pl.col('strand1_blocks')
        ).then(
            pl.lit("+")
        ).otherwise(
            pl.lit("-")
        ).alias('strand_1_relocated')
    ).filter(
        (
            pl.when(
                pl.col('strand1_blocks') == '+'
            ).then(
                pl.col('end_source_blocks') - pl.col('pos1_pairs') >= old_border_shift
            ).otherwise(
                pl.col('pos1_pairs') - pl.col('start_source_blocks') >= old_border_shift
            )
        ) & (
            pl.when(
                pl.col('strand_1_relocated') == '+'
            ).then(
                pl.col('end_target_pairs') - pl.col('pos_1_relocated') >= new_border_shift
            ).otherwise(
                pl.col('pos_1_relocated') - pl.col('start_target_pairs') >= new_border_shift
            )
        )
    ).select(
        pl.col('pos_1_relocated'),
        pl.col('strand_1_relocated'),
        pl.col('chrom_target_pairs').alias('chrom_1_relocated'),

        pl.col('readid_blocks').alias('readid'),
        pl.col('pair_type_blocks').alias('pair_type'),
        pl.col('mapq1_blocks').alias('mapq1'),
        pl.col('mapq2_blocks').alias('mapq2'),

        pl.col('strand2_blocks').alias('strand2'),
        pl.col('chrom2_blocks').alias('chrom2'),
        pl.col('pos2_blocks').alias('pos2')
    )


    overlapped_target_pairs = pb.overlap(
        df1=overlapped_source_pairs.with_columns(
            pl.col('pos2').alias('pos_2_end')
        ),
        df2=chains.blocks.select(
                pl.col("name_chain_source").alias("chrom_source"),
                pl.col("start_aln_source").alias("start_source"),
                pl.col("end_aln_source").alias("end_source"),
                pl.col("name_chain_target").alias("chrom_target"),
                pl.col("start_aln_target").alias("start_target"),
                pl.col("end_aln_target").alias("end_target"),
                pl.col("strand_chain_target"),
                pl.col("strand_chain_source")
            ),
        suffixes=('_pairs', '_blocks'),
        cols2=('chrom2', 'pos2', 'pos_2_end'),
        cols1=('chrom_source', 'start_source', 'end_source'),
        output_type='polars.DataFrame'
    ).with_columns(
        pl.when(
            pl.col('strand_chain_target_pairs') == '+'
        ).then(
            pl.col('end_target_pairs') - (pl.col('end_source_blocks') - pl.col('pos2_pairs'))
        ).otherwise(
            pl.col('start_target_pairs') + (pl.col('end_source_blocks') - pl.col('pos2_pairs'))
        ).alias('pos_2_relocated'),

        pl.when(
            pl.col('strand_chain_target_pairs') == pl.col('strand2_blocks')
        ).then(
            pl.lit("+")
        ).otherwise(
            pl.lit("-")
        ).alias('strand_2_relocated')
    ).filter(
        (
            pl.when(
                pl.col('strand2_blocks') == '+'
            ).then(
                pl.col('end_source_blocks') - pl.col('pos2_pairs') >= old_border_shift
            ).otherwise(
                pl.col('pos2_pairs') - pl.col('start_source_blocks') >= old_border_shift
            )
        ) & (
            pl.when(
                pl.col('strand_2_relocated') == '+'
            ).then(
                pl.col('end_target_pairs') - pl.col('pos_2_relocated') >= new_border_shift
            ).otherwise(
                pl.col('pos_2_relocated') - pl.col('start_target_pairs') >= new_border_shift
            )
        )
    ).select(
        pl.col('readid_blocks').alias('readid'),
        pl.col('chrom_1_relocated_blocks').alias('chrom1'),
        pl.col('pos_1_relocated_blocks').alias('pos1'),
        pl.col('chrom_target_pairs').alias('chrom2'),
        pl.col('pos_2_relocated').alias('pos2'),
        pl.col('strand_1_relocated_blocks').alias('strand1'),
        pl.col('strand_2_relocated').alias('strand2'),
        pl.col('pair_type_blocks').alias('pair_type'),
        pl.col('mapq1_blocks').alias('mapq1'),
        pl.col('mapq2_blocks').alias('mapq2')
    )
    
    return overlapped_target_pairs