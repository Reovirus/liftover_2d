import numpy as np
import cooler
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

from typing import Iterator, Literal, Iterable, List

import json

from tests.benchmark.dist_utils.compute_sets import (
    compute_cols,
    compute_diags,
    compute_mats
)

from tests.benchmark.dist_utils.samle_cnts import (
    sample_counts_ram
)

def create_sample_json_ram(
    cooler_file: cooler.Cooler,
    keep_rates: Iterable[float]=np.arange(0.1, 1.0, 0.05),
    random_state: int=42,
    return_strategy: Literal['total', 'per_chromosome']='total',
    repeats: int=100,
    metric_list: List[Literal['kl', 'wasserstein', 'mse', 'frobenius']]=['kl', 'mse'],
    alias: str='sampled',
    compute_metrics: List[Literal['absolute', 'relative']]=['absolute', 'relative']
) -> Iterator[dict]:
    for samplingrate in keep_rates:
        for sampled, original, chrom, repeat in sample_counts_ram(
            clr_file=cooler_file, 
            keep_rate=samplingrate, 
            random_state=random_state,
            return_strategy=return_strategy,
            repeats=repeats
        ):
            ans = {
                'chrom': chrom,
                'repeat': repeat,
                'alias': alias,
                'sampling_rate': samplingrate
            }
            if 'relative' in compute_metrics:
                rate_matrix = (sampled + 1)/(original+1)
                ideal_result = np.ones(original.shape)
                mtrs = {
                    'diags_relative': compute_diags(rate_matrix, ideal_result, metric_list=metric_list),
                    'cols_relative': compute_cols(rate_matrix, ideal_result, metric_list=metric_list),
                    'mats_relative': compute_mats(rate_matrix, ideal_result, metric_list=metric_list)
                }
                ans.update(mtrs)
            if 'absolute' in compute_metrics:
                mtrs = {
                    'diags_absolute': compute_diags(original, sampled, metric_list=metric_list),
                    'cols_absolute': compute_cols(original, sampled, metric_list=metric_list),
                    'mats_absolute': compute_mats(original, sampled, metric_list=metric_list)
                }
                ans.update(mtrs)
            yield ans

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)
        

def save_dists_json(
    dists: List[dict],
    out_json: str,
    indent: int=4
):
    with open(out_json, 'w') as f:
        json.dump(dists, f, cls=NumpyArrayEncoder, indent=indent)