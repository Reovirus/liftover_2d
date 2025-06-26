from typing import List, Dict, Any, BinaryIO, TextIO, Union
import gzip
import polars as pl

from src._utils import (
    TYPE_POLARS_INTEGER,
    TYPE_POLARS_STRING
)


_CHAIN_SCHEMA = {
    'title': TYPE_POLARS_STRING, 
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
    'index_chain': TYPE_POLARS_INTEGER, 
    'index_chain_block': TYPE_POLARS_INTEGER 
}

_block_cols = ["source_aln_start", "source_aln_end", "target_aln_start", "size_aln", "gap_aln_source", "gap_aln_target", "index_chain_block"]
_BLOCK_SCHEMA = { col : TYPE_POLARS_INTEGER for col in _block_cols}


class ChainReader:
    def __init__(self, chain_file: Union[BinaryIO, TextIO, str]):
        self.__chain_file = chain_file
        self.__read_chain()

    #its dangerous to load the full chain file twice. add economy and slow mode with rows iterator
    def __read_chain(self):
        if isinstance(self.__chain_file, str):
            with gzip.open(self.__chain_file, 'rb') as f:
                file_content = f.read()
        else:
            file_content = self.__chain_file.read()

        lines = file_content.splitlines()
        chains_data = b''
        blocks_data = b''

        i = 0
        index_chain_block = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith(b'#') or line.strip() == b"":
                i += 1
                continue

            if line.startswith(b'chain'):
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
                
                index_chain_block += 1

            else:
                i += 1
        chains_df = pl.read_csv(
                        chains_data,
                        separator=" ",
                        has_header=False,
                        schema=_CHAIN_SCHEMA
                    )
        blocks_df = pl.read_csv(
                        blocks_data,
                        separator="\t",
                        has_header=False,
                        schema=_BLOCK_SCHEMA
                    ).fill_null(0)

        blocks_df = blocks_df.join(chains_df, on='index_chain_block')

        blocks_df = blocks_df.with_columns(
            pl.col("size_aln").shift(-1).over("index_chain_block").fill_null(0).alias("size_aln"),
        )

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

        self.__chains = chains_df
        self.__blocks = blocks_df


    @property
    def blocks(self):
        return self.__blocks

    @blocks.setter
    def value(self, new_value):
        raise ValueError("Blocks cannot be set directly.")
    
    @property
    def chains(self):
        return self.__chains

    @chains.setter
    def value(self, new_value):
        raise ValueError("Chains cannot be set directly.")
    

    def chain_to_cheme(self, target_bins, source_bins, return_chroms=False):
        pass