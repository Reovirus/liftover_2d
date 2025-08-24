import cooler
import numpy as np

from typing import Iterator, Literal, Tuple

def sample_counts_ram(
        clr_file: cooler.Cooler, 
        keep_rate: float, 
        random_state=42,
        return_strategy: Literal['total', 'per_chromosome']='total',
        repeats: int=100
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, str, int]]:
    '''
    Samples counts from cooler file in RAM.
    :param clr_file: cooler.Cooler object to sample from
    :param keep_rate: Probability of keeping each count (for binomial sampling)
    :param random_state: Random seed for reproducibility
    :param return_strategy: 'total' to sample from the whole matrix, 'per_chromosome' to sample per chromosome
    :param repeats: Number of times to repeat the sampling
    :return: Iterator of tuples (sampled_matrix, original_matrix, chromosome_name, repeat_number)
    '''
    if return_strategy == 'per_chromosome':
        chrom_dict = {}
        for cromname in clr_file.chromnames:
            chrom_dict[cromname] = clr_file.matrix(balance=False).fetch(cromname)[:]
    elif return_strategy == 'total':
        chrom_dict = {}
        chrom_dict['__total__'] = clr_file.matrix(balance=False)[:]
    else:
        raise ValueError("return_strategy must be 'total' or 'per_chromosome'")
    
    for repeat_num in range(repeats):
        for key, matrix in chrom_dict.items():
            rng = np.random.default_rng(random_state+repeat_num)
            yield rng.binomial(n=matrix, p=keep_rate), matrix, key, repeat_num


def sample_counts(
        clr_file: cooler.Cooler, 
        keep_rate: float, 
        random_state=42,
        return_strategy: Literal['total', 'per_chromosome']='total'
    ) -> Iterator[np.ndarray]:
    if return_strategy == 'per_chromosome':
        for i in clr_file.chromnames:
            rng = np.random.default_rng(random_state)
            yield rng.binomial(n=clr_file.matrix(balance=False).fetch(i)[:], p=keep_rate), i
    elif return_strategy == 'total':
        rng = np.random.default_rng(random_state)
        yield rng.binomial(n=clr_file.matrix(balance=False)[:], p=keep_rate), '__total__'
    else:
        raise ValueError("return_strategy must be 'total' or 'per_chromosome'")