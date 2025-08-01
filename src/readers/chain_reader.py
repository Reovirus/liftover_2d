from typing import List, Dict, Any, BinaryIO, TextIO, Union
import gzip
import polars as pl
import io

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

_block_pl_cols = ['index_chain_block', "size_aln",  "gap_aln_source", "gap_aln_target"]
_BLOCK_PL_SCHEMA = { col : TYPE_POLARS_INTEGER for col in _block_pl_cols}

class ChainReader:
    def __init__(self, chain_file: Union[BinaryIO, TextIO, str], use_buffered: bool = False, buffer_size: int = 4 * 1024 * 1024):
        self.__chain_file = chain_file
        if use_buffered:
            self.__read_chain_buffered(buffer_size)
        else:
            self.__read_chain()

    def __read_chain_buffered(self, buffer_size: int = 4 * 1024 * 1024):
        chains_lines = []
        blocks_lines = []
        index_chain_block = -1
        is_block = False

        if isinstance(self.__chain_file, str):
            gz = gzip.open(self.__chain_file, "rb")
        else:
            gz = self.__chain_file 
        buf = io.BufferedReader(gz, buffer_size=buffer_size)

        for raw in buf:
            line = raw.rstrip(b"\r\n")
            if not line or line.startswith(b"#"):
                continue

            if line.startswith(b"chain"):
                index_chain_block += 1
                is_block = True
                chains_lines.append(line + b" " + str(index_chain_block).encode())
                blocks_lines.append(f"{index_chain_block}\t0\t0\t0".encode())
                continue

            if is_block:
                if line.startswith(b"chain") or line.startswith(b"#"):
                    is_block = False
                    continue
                blocks_lines.append(str(index_chain_block).encode() + b"\t" + line)

        chains_df = pl.DataFrame({"raw": chains_lines}).with_columns(
            pl.col(
                "raw"
            ).cast(pl.String).str.split_exact(
                " ", len(_CHAIN_SCHEMA.keys()) - 1
            ).struct.rename_fields(
                list(_CHAIN_SCHEMA.keys())
            ).alias("blk")
        ).unnest(
            "blk"
        ).cast(_CHAIN_SCHEMA)

        blocks_df = pl.DataFrame({"raw": blocks_lines}).with_columns(
            pl.col(
                "raw"
            ).cast(pl.String).str.split_exact(
                "\t", len(_BLOCK_PL_SCHEMA.keys()) - 1
            ).struct.rename_fields(
                list(_BLOCK_PL_SCHEMA.keys())
            ).alias("blk")
        ).unnest(
            "blk"
        ).cast(_BLOCK_PL_SCHEMA).fill_null(0)

        blocks_df.drop_in_place("raw")
        chains_df.drop_in_place("raw")

        blocks_df = blocks_df.join(chains_df, on="index_chain_block")
        blocks_df = blocks_df.with_columns(
            pl.col("size_aln").shift(-1).over("index_chain_block").fill_null(0).alias("size_aln")
        )
        blocks_df = blocks_df.with_columns(
            (pl.col("size_aln").cum_sum().over("index_chain_block")
             + pl.col("gap_aln_source").cum_sum().over("index_chain_block")
             + pl.col("start_chain_source")).alias("end_aln_source"),
            (pl.col("size_aln").cum_sum().over("index_chain_block")
             + pl.col("gap_aln_target").cum_sum().over("index_chain_block")
             + pl.col("start_chain_target")).alias("end_aln_target")
        )
        blocks_df = blocks_df.with_columns(
            (pl.col("end_aln_source") - pl.col("size_aln")).alias("start_aln_source"),
            (pl.col("end_aln_target") - pl.col("size_aln")).alias("start_aln_target")
        )
        blocks_df = blocks_df.filter(pl.col("size_aln") > 0)
        blocks_df = blocks_df.select([
            'name_chain_source', 'start_aln_source', 'end_aln_source', 'strand_chain_source',
            'name_chain_target', 'start_aln_target', 'end_aln_target', 'strand_chain_target',
            'size_aln', 'start_chain_source', 'end_chain_source', 'size_chain_source',
            'start_chain_target', 'end_chain_target', 'size_chain_target',
            'score', 'index_chain', 'index_chain_block'
        ])

        self.__chains = chains_df
        self.__blocks = blocks_df


    def __read_chain(self):
        if isinstance(self.__chain_file, str):
            with gzip.open(self.__chain_file, 'rb') as f:
                file_content = f.read()
        else:
            file_content = self.__chain_file.read()

        lines = file_content.splitlines()
        chains_data = []
        blocks_data = []

        i = 0
        index_chain_block = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith(b'#') or line.strip() == b"":
                i += 1
                continue

            if line.startswith(b'chain'):
                header_line = line
                chains_data.append(header_line + f' {index_chain_block}'.encode('utf-8'))
                i += 1

                blocks_data.append(f'{index_chain_block}\t'.encode('utf-8') + b'0\t0\t0')
                while i < len(lines):
                    line = lines[i]
                    if line.startswith(b'chain') or line.startswith(b'#') or line.strip() == b"":
                        break
                    blocks_data.append(f'{index_chain_block}\t'.encode('utf-8') + line)
                    i += 1
                
                index_chain_block += 1

            else:
                i += 1

        chains_df = pl.DataFrame({"raw": chains_data}).with_columns(
            pl.col(
                "raw"
            ).cast(pl.String).str.split_exact(
                " ", len(_CHAIN_SCHEMA.keys()) - 1
            ).struct.rename_fields(
                list(_CHAIN_SCHEMA.keys())
            ).alias("blk")
        ).unnest(
            "blk"
        ).cast(_CHAIN_SCHEMA)

        blocks_df = pl.DataFrame({"raw": blocks_data}).with_columns(
            pl.col(
                "raw"
            ).cast(pl.String).str.split_exact(
                "\t", len(_BLOCK_PL_SCHEMA.keys())
            ).struct.rename_fields(
                list(_BLOCK_PL_SCHEMA.keys())
            ).alias("blk")
        ).unnest(
            "blk"
        ).cast(_BLOCK_PL_SCHEMA).fill_null(0)

        blocks_df.drop_in_place("raw")
        chains_df.drop_in_place("raw")

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
    

def read_chain():
    pass

def read_chain_buffered():
    pass



def read_chain_file(file: Union[BinaryIO, TextIO, str], use_buffered: bool = False, buffer_size: int = 4 * 1024 * 1024) -> ChainReader:
    """
    Reads a chain file and returns a ChainReader object.
    
    :param file: Path to the chain file or a file-like object.
    :param use_buffered: Whether to read the file using buffered reading.
    :param buffer_size: Size of the buffer for buffered reading.
    :return: ChainReader object containing chains and blocks data.
    """

    if isinstance()
    return ChainReader(file, use_buffered=use_buffered, buffer_size=buffer_size)