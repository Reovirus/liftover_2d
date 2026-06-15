import gzip
import polars as pl
from io import BytesIO

import cooler
import numpy as np
import pandas as pd
import tqdm
from cooler.core import (
    CSRReader,
    DirectRangeQuery2D,
)

import bioframe

TYPE_POLARS_INTEGER = pl.UInt64
TYPE_POLARS_SIGNED_INTEGER = pl.Int64
TYPE_POLARS_STRING  = pl.String
TYPE_POLARS_FLOAT   = pl.Float64
POLARS_VMAX = pl.DataFrame({'vmax':0}).with_columns( vmax=(pl.UInt64.max()//2) ).to_numpy()[0, 0]

block_pl_cols = ['index_chain_block', "size_aln",  "gap_aln_source", "gap_aln_target"]
block_pl_schema = { col : TYPE_POLARS_INTEGER for col in block_pl_cols}

block_cols = ["source_aln_start", "source_aln_end", "target_aln_start", "size_aln", 'index_chain_block']
block_schema = { col : TYPE_POLARS_INTEGER for col in block_cols}

chain_cols = ['title', 'score', 
              'name_chain_source', 'size_chain_source', 'strand_chain_source', 'start_chain_source', 'end_chain_source',
              'name_chain_target', 'size_chain_target', 'strand_chain_target', 'start_chain_target', 'end_chain_target',
              'index_chain', 'index_chain_block']

chain_schema = {
    'title': TYPE_POLARS_STRING, # Most linkely simply "chain" in the input file
    'score' : TYPE_POLARS_INTEGER,
    'name_chain_source': TYPE_POLARS_STRING,
    'size_chain_source' : TYPE_POLARS_INTEGER,
    'strand_chain_source': TYPE_POLARS_STRING,
    'start_chain_source': TYPE_POLARS_INTEGER,
    'end_chain_source': TYPE_POLARS_INTEGER,
    'name_chain_target': TYPE_POLARS_STRING, 
    'size_chain_target': TYPE_POLARS_INTEGER, 
    'strand_chain_target': TYPE_POLARS_STRING,
    'start_chain_target': TYPE_POLARS_INTEGER, 
    'end_chain_target': TYPE_POLARS_INTEGER, 
    'index_chain': TYPE_POLARS_INTEGER, # As reported in the chain file
    'index_chain_block': TYPE_POLARS_INTEGER # Order of the chain block in the file for indexing
}

chunk_schema = {'bin1_id':TYPE_POLARS_INTEGER, 
                'bin2_id':TYPE_POLARS_INTEGER,
                'count': TYPE_POLARS_INTEGER,
                }

chunk_annotated_schema = {
    'bin1_id':TYPE_POLARS_INTEGER, 
    'bin2_id':TYPE_POLARS_INTEGER,
    'count': TYPE_POLARS_INTEGER,
    'chrom1': TYPE_POLARS_STRING, 
    'start1': TYPE_POLARS_INTEGER,
    'end1': TYPE_POLARS_INTEGER,
    'weight1': TYPE_POLARS_FLOAT,
    'chrom2': TYPE_POLARS_STRING, 
    'start2': TYPE_POLARS_INTEGER,
    'end2': TYPE_POLARS_INTEGER,
    'weight2': TYPE_POLARS_FLOAT,
}


def read_chain_file(input_file):
    """
    Reads a UCSC chain file and parses it into DataFrames for chains and blocks.

    Note:
        UCSC chain files account for the '-' strand orientation internally.
        "When the strand value is '-', position coordinates are listed in terms of the reverse-complemented sequence."
        (https://genome.ucsc.edu/goldenpath/help/chain.html)
        However, we may need to take that into account when converting the coordinates within each alignment block.

    Args:
        input_file (str): Path to the chain file (gzipped).

    Returns:
        tuple: (chains_df, blocks_df)
            chains_df (pl.DataFrame): DataFrame containing chain header information.
            blocks_df (pl.DataFrame): DataFrame containing alignment blocks.
    """
    # Read the entire chain file content
    with gzip.open(input_file, 'rb') as f:
        file_content = f.read()

    # Split content into lines
    lines = file_content.splitlines()
    chains_data = b''
    blocks_data = b''

    i = 0
    index_chain_block = 0
    while i < len(lines):
        line = lines[i]

        # Skip comments and empty lines
        if line.startswith(b'#') or line.strip() == b"":
            i += 1
            continue

        if line.startswith(b'chain'):
            # Process chain header line and append index_chain_block
            header_line = line
            chains_data += header_line + f' {index_chain_block}\n'.encode('utf-8')
            i += 1

            blocks_data += f'{index_chain_block}\t'.encode('utf-8') + b'0\t0\t0\n'
            while i < len(lines):
                line = lines[i]
                if line.startswith(b'chain') or line.startswith(b'#') or line.strip() == b"":
                    break
                blocks_data += f'{index_chain_block}\t'.encode('utf-8') + line + b'\n'
                i += 1
            
            # Increment index_chain_block for the next chain
            index_chain_block += 1

        else:
            i += 1
    # Convert lists to strings and read into DataFrames
    chains_df = pl.read_csv(
                    chains_data,
                    separator=" ",
                    has_header=False,
                    schema=chain_schema
                )
    blocks_df = pl.read_csv(
                    blocks_data,
                    separator="\t",
                    has_header=False,
                    schema=block_pl_schema
                ).fill_null(0)

    blocks_df = blocks_df.join(chains_df, on='index_chain_block')

    # Shift the alignment size for appropriate cumsum calculation later:
    blocks_df = blocks_df.with_columns(
        pl.col("size_aln").shift(-1).over("index_chain_block").fill_null(0).alias("size_aln"),
    )

    # Calculate the positions of the alignments from the gaps and information about chain starts:
    blocks_df = blocks_df.with_columns(
        (pl.col("size_aln").cum_sum().over("index_chain_block") + \
         pl.col("gap_aln_source").cum_sum().over("index_chain_block") + \
         pl.col("start_chain_source")).alias('end_aln_source'),
        (pl.col("size_aln").cum_sum().over("index_chain_block") + \
         pl.col("gap_aln_target").cum_sum().over("index_chain_block") + \
        pl.col("start_chain_target")).alias('end_aln_target'),
                    )
    blocks_df = blocks_df.with_columns(
        (pl.col('end_aln_source') - pl.col("size_aln")).alias('start_aln_source'),   
        (pl.col('end_aln_target') - pl.col("size_aln")).alias('start_aln_target'),   
                    )

    blocks_df = blocks_df.filter(pl.col("size_aln")>0)

    blocks_df = blocks_df.select([
        'name_chain_source', "start_aln_source", 'end_aln_source', 'strand_chain_source', 
        'name_chain_target', "start_aln_target", "end_aln_target", 'strand_chain_target', 
        "size_aln",
        'start_chain_source', 'end_chain_source', 'size_chain_source', 
        'start_chain_target', 'end_chain_target', 'size_chain_target',
        'score',
        'index_chain', 'index_chain_block'
        ])

    return chains_df, blocks_df



def chain_to_scheme(bins_source, bins_target, chains_df, blocks_df, return_chroms=False):
    """
    Generates a conversion scheme from source bins to target bins using chain alignments.

    Parameters:
        bins_source (pd.DataFrame): Bins from the source genome.
        bins_target (pd.DataFrame): Bins from the target genome.
        chains_df (pl.DataFrame): DataFrame of chain information.
        blocks_df (pl.DataFrame): DataFrame of block information.

    Returns:
        tuple: (conversion_scheme, bins_right_pl)
            conversion_scheme (pl.DataFrame): Mapping from source bins to target bins with splitting fractions.
            bins_right_pl (pl.DataFrame): Detailed mapping information.
    """
    
    # Run bioframe overlap in pandas.
    # Note that this will fragment the bins if they overlap the chains and alignments in the input file:
    blocks_df_pd = blocks_df.to_pandas()

    bins_overlapped_left = bioframe.overlap( bins_source, blocks_df_pd, how='left', 
        cols1=('chrom', 'start', 'end'), 
        cols2=('name_chain_source', 'start_aln_source', 'end_aln_source'), 
        return_input=True, return_index=True, return_overlap=True, suffixes=('_bin_source', ''))\
    .rename({'overlap_start':'start_source_overlap', 'overlap_end': 'end_source_overlap'}, axis=1)

    bins_left_pl = pl.from_pandas( bins_overlapped_left )

    bins_left_pl = bins_left_pl.with_columns( (pl.col('end_source_overlap')-pl.col('start_source_overlap')).alias("size_source_overlap") )

    # Determine the positions of the fragments in the target assembly:
    bins_left_pl = bins_left_pl.with_columns(
        
        # Regions starts in the new coordinates: 
        pl.when(pl.col("strand_chain_target") == '+')
            .then(      pl.col("start_aln_target") + (pl.col("start_source_overlap") - pl.col("start_aln_source")) )                                 # Forward chain
            .otherwise( pl.col("end_aln_target")   - (pl.col("start_source_overlap") - pl.col("start_aln_source")) - pl.col("size_source_overlap") ) # Reverse complement, we'll always report smallset coordinate first
            .cast(TYPE_POLARS_INTEGER)
            .alias("start_bin_source_converted"),

        # Regions ends in the new coordinates: 
        pl.when(pl.col("strand_chain_target") == '+')
            .then(     pl.col("start_aln_target") +  pl.col("start_source_overlap") - pl.col("start_aln_source") + pl.col("size_source_overlap") )
            .otherwise(pl.col("end_aln_target")   - (pl.col("start_source_overlap") - pl.col("start_aln_source")) )
            .cast(TYPE_POLARS_INTEGER)
            .alias("end_bin_source_converted"),

    )

    bins_left_pl = bins_left_pl.select(['index_bin_source', 'chrom_bin_source', 'start_bin_source', 'end_bin_source', # Information about the bins of the source genome
                        'start_source_overlap', 'end_source_overlap', 'size_source_overlap',           # Fragments of the bins of the source gneome, split by the boundaries of the chains and alignments within them
                        'name_chain_target', 'start_bin_source_converted', 'end_bin_source_converted', # Fragments of the bins of the source genome with coordinates converted into the new genome
                        "start_aln_source", "end_aln_source", "start_aln_target", "end_aln_target", "strand_chain_target" # Info about alignment
                                    ])

    # Convert the scheme to pandas:
    bins_left_pd = bins_left_pl.drop_nulls().to_pandas()

    # Overlap the source bins with the target bins, knowing the coordinates of the bin fragments in the new genome:
    bins_overlapped_right = bioframe.overlap( bins_target, bins_left_pd, how='right',
                                            cols1=('chrom', 'start', 'end'), 
                                            cols2=('name_chain_target', 'start_bin_source_converted', 'end_bin_source_converted'), 
                                            return_input=True, return_index=True, return_overlap=True, suffixes=('_bin_target', '')) \
        .rename({
            'overlap_start': 'start_target_overlap', 
            'overlap_end': 'end_target_overlap'
            }, axis=1)

    # Convert back to Polars DataFrame.
    schema_upd = {
        'index_bin_source':TYPE_POLARS_INTEGER,
        'index_bin_target':TYPE_POLARS_INTEGER,
        'index':TYPE_POLARS_INTEGER,
        'chrom_bin_target':TYPE_POLARS_STRING,
        'start_bin_target':TYPE_POLARS_INTEGER,
        'end_bin_target':TYPE_POLARS_INTEGER,
        'weight_bin_target':TYPE_POLARS_FLOAT,
        'chrom_bin_source':TYPE_POLARS_STRING,
        'start_bin_source':TYPE_POLARS_INTEGER,
        'end_bin_source':TYPE_POLARS_INTEGER,
        'start_target_overlap':TYPE_POLARS_INTEGER,
        'end_target_overlap':TYPE_POLARS_INTEGER,    
    }
    bins_right_pl = pl.from_pandas(bins_overlapped_right).cast(schema_upd)

    bins_right_pl = bins_right_pl.with_columns(
        pl.when(pl.col("start_target_overlap").is_not_null())
        .then( pl.col("end_target_overlap")-pl.col("start_target_overlap") )
        .otherwise(pl.col("size_source_overlap"))
        .alias("size_target_overlap")
    )

    bins_right_pl = bins_right_pl.with_columns(
        # How the initial bins are split into regions (proportions):
        (pl.col("size_target_overlap") / pl.col("size_target_overlap").sum().over("index_bin_source")).alias("split_into_fractions")
    )

    if return_chroms:
        conversion_scheme = bins_right_pl.select([
            'index_bin_source', 'index_bin_target', 'chrom_bin_target', 'split_into_fractions'
        ]).sort('index_bin_source', 'index_bin_target')
    else:
        conversion_scheme = bins_right_pl.select([
            'index_bin_source', 'index_bin_target', 'split_into_fractions'
        ]).sort('index_bin_source', 'index_bin_target')

    # TODO: make sure that if we've lost part of the initial target bin due to first conversion (it's unmapped in target assembly),
    # we still allow to take it into account later - e.g. throw away proportionate values from the bin counts.

    return conversion_scheme, bins_right_pl


## TODO: separate the steps of data reading and conversion.
# Add the function to handle the input data
# Add the function to handle pairs 

def convert_cooler(clr, conversion_scheme, chunksize=1_000_000, mode='resample', cvd_norm=None, random_seed=None):
    """
    Converts a cooler file using a provided conversion scheme.

    Parameters:
        clr (cooler.Cooler): Cooler object to be converted.
        conversion_scheme (pl.DataFrame): DataFrame mapping source bins to target bins with splitting fractions.
        chunksize (int): Number of pixels to process in each chunk.
        mode (str): 'resample' for multinomial resampling (slower), 'proportional' for proportional counts (faster).
        cvd_norm (indexed array): contact-vs-distance binned distancs vs contact probability (None by default).
    Yields:
        pd.DataFrame: DataFrames of converted chunks ready to be aggregated and saved.
    """
    
    n_bins = clr.info['nbins']

    # Dump everything, not selecting by regions:
    bbox = (0, n_bins, 0, n_bins)

    h5 = clr.open("r")
    reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
    field = "count"

    engine = DirectRangeQuery2D(reader, field, bbox, chunksize)

    chunks = (
        pl.from_pandas(
            pd.DataFrame(
                dct,
                columns=["bin1_id", "bin2_id", field],
            )
        ).cast(chunk_schema)
        for dct in engine
    )

    if random_seed is not None: 
        np.random.seed(random_seed)

    for chunk_pl in tqdm.tqdm(chunks):

        # Swap values so that bin1_id is always smaller bin2_id:
        chunk_pl = chunk_pl.with_columns(
            
            pl.when(pl.col("bin1_id") > pl.col("bin2_id"))
            .then(pl.col("bin2_id"))
            .otherwise(pl.col("bin1_id"))
            .alias("bin1_id"),

            pl.when(pl.col("bin1_id") > pl.col("bin2_id"))
            .then(pl.col("bin1_id"))
            .otherwise(pl.col("bin2_id"))
            .alias("bin2_id") 
        )

        # Group by and aggregate counts
        chunk_pl = chunk_pl.group_by("bin1_id", "bin2_id").agg(pl.col('count').sum())

        # Annotate by the conversion scheme (each bin might be split into multiple target bins with different coverages)
        for i in ['1', '2']:
            chunk_pl = chunk_pl.join(conversion_scheme, 
                right_on='index_bin_source', 
                left_on=f'bin{i}_id', 
                how='left').rename({
                'index_bin_target' : f'index_bin_target_{i}', 
                'split_into_fractions' : f'split_into_fractions_{i}'},
                )
            if cvd_norm is not None: 
                chunk_pl = chunk_pl.rename({
                    'chrom_bin_target' : f'chrom_bin_target_{i}'
                })

        chunk_pl = chunk_pl.with_columns(
            pl.col("split_into_fractions_1").fill_null(1.0),
            pl.col("split_into_fractions_2").fill_null(1.0),
            pl.col("count").fill_null(0.0),
        )

        vmax = max( chunk_pl['bin1_id'].max(), chunk_pl['bin2_id'].max() )
        assert vmax < POLARS_VMAX//2, \
            "Bin ids are too large to use this library safely at the moment; we still need to implement the scheme for hashing these large numbers that you have"

        chunk_pl = chunk_pl.with_columns(
            pl.col("split_into_fractions_1").fill_null(1.0),
            pl.col("split_into_fractions_2").fill_null(1.0),
            (pl.col('bin1_id').cast(TYPE_POLARS_INTEGER) + vmax*pl.col('bin2_id').cast(TYPE_POLARS_INTEGER) ).alias('bin_hash')
        ).sort('bin_hash')

        # Normalize probabilities by the contact probabilities at a distance:
        if cvd_norm is not None: 

            cvd_cis, cvd_trans = cvd_norm
            cvd_cis = dict(cvd_cis)

            assert 'chrom_bin_target_1' in chunk_pl.columns, 'Please, provide conversion schema with chrom_bin_target as input'

            chunk_pl = chunk_pl.with_columns(
                (pl.col('index_bin_target_1').cast(TYPE_POLARS_SIGNED_INTEGER) - pl.col('index_bin_target_2').cast(TYPE_POLARS_SIGNED_INTEGER)).abs().alias('bin_distance')
            )
            chunk_pl = chunk_pl.with_columns(
                pl.when(pl.col("chrom_bin_target_1") == pl.col("chrom_bin_target_2"))
                    .then( pl.col('bin_distance').replace_strict(cvd_cis, default=cvd_trans) ) # We assume default trans levels for the contacts that are very far apart
                    .otherwise(cvd_trans)
                    .alias('prob_from_distance'),
                
            )
            chunk_pl = chunk_pl.with_columns(
                (pl.col("split_into_fractions_1") * pl.col("split_into_fractions_2")).alias('prob_no_norm'),
                (pl.col("split_into_fractions_1") * pl.col("split_into_fractions_2") * pl.col('prob_from_distance')).alias('prob')
            )
            # print(chunk_pl)
            # break

        # Get the probabilities without normalizing by distance but only the fractions of bin falling into different output bins
        else: 
            chunk_pl = chunk_pl.with_columns(
                (pl.col("split_into_fractions_1") * pl.col("split_into_fractions_2")).alias('prob')
            )

        if mode=='resample':
            # Very fast numpy-based replacement of the groupby:
            hashed_bins = chunk_pl['bin_hash'].to_numpy()
            probs = chunk_pl['prob'].to_numpy()
            ns = chunk_pl['count'].to_numpy()

            result = []
            probs_group = []
            hash_prev = hashed_bins[0]
            for i in range(len(hashed_bins)):
                v_hash = hashed_bins[i]
                
                if v_hash!=hash_prev:
                    # Write down the previous block:
                    if i==0:
                        continue
                    else:
                        probs_group = np.array(probs_group)
                        probs_group /= probs_group.sum()

                        result_group = np.random.multinomial(n=n, pvals=probs_group)
                        result += [result_group]
                        
                    # Renew the probs block:
                    probs_group = [probs[i]]
                
                else:
                    # Append to the probs block:
                    probs_group += [probs[i]]

                n = ns[i]
                hash_prev = v_hash

            # Final element:
            probs_group = np.array(probs_group)
            result_group = np.random.multinomial(n=n, pvals=probs_group)
            result += [result_group]

            result = np.concatenate(result)

            chunk_pl = chunk_pl.with_columns(count_sampled=result)

        else:
            chunk_pl = chunk_pl.with_columns(
                (pl.col('split_into_fractions_1')*pl.col('split_into_fractions_2')*pl.col("count")).alias("count_sampled")
            )

        # Final cleanup and aggregation
        chunk_pl = chunk_pl.drop_nulls().select([
            pl.col("index_bin_target_1").alias("bin1_id"),
            pl.col("index_bin_target_2").alias("bin2_id"),
            pl.col("count_sampled").alias("count"),
        ]).cast({
            "bin1_id": TYPE_POLARS_INTEGER,
            "bin2_id": TYPE_POLARS_INTEGER,
            "count": TYPE_POLARS_INTEGER,
        })

        # Ensure bin1_id <= bin2_id and aggregate again and aggregate
        chunk_pl = chunk_pl.with_columns(
            
            pl.when(pl.col("bin1_id") > pl.col("bin2_id"))
            .then(pl.col("bin2_id"))
            .otherwise(pl.col("bin1_id"))
            .alias("bin1_id"),

            pl.when(pl.col("bin1_id") > pl.col("bin2_id"))
            .then(pl.col("bin1_id"))
            .otherwise(pl.col("bin2_id"))
            .alias("bin2_id") 
        ).group_by('bin1_id', 'bin2_id', maintain_order=True).agg(pl.sum("count"))

        # Correctness checkup:
        chunk = chunk_pl.sort('bin1_id', 'bin2_id').to_pandas()
        
        yield(chunk)
