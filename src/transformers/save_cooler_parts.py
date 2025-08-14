import tempfile
import shutil
import atexit
import time
import os
import threading
import cooler
import uuid

from datetime import datetime
from typing import Iterable

from src.readers import CoolerPolars


def cooler_saver(cooler_polars_iterator: Iterable[CoolerPolars], res_path: str, ttl_seconds: int = 2 * 60 * 60, print_progress: bool = True, mergebuff=1_000_000) -> cooler.Cooler:
    '''
    :param cooler_polars_iterator: Iterable of CoolerPolars objects to save and merge
    :param res_path: Path to save the merged cooler file
    :param ttl_seconds: Time to live for temporary files, after which they will be deleted
    :return: Cooler object from the merged file
    Saves multiple CoolerPolars from iterator into many tmps and merges them into one cooler file.
    If falls, tmp files will be deleted after ttl_seconds.
    If success, tmp files will be deleted immediately.
    Returns Cooler object from merged file.
    '''

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    prefix = f"cooler_tmp_{ts}_{suffix}_"
    
    tmp_dir = tempfile.mkdtemp(prefix=prefix)

    if print_progress:
        print(f"[INFO] Created temporary directory: {tmp_dir}")

    def cleanup():
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            if print_progress:
                print(f"[INFO] Deleted temporary directory: {tmp_dir}")

    atexit.register(cleanup)
    threading.Thread(target=lambda: (time.sleep(ttl_seconds), cleanup()), daemon=True).start()

    part_paths = []
    for i, cooler_polars in enumerate(cooler_polars_iterator):
        part_path = os.path.join(tmp_dir, f"part_{i}.cool")
        cooler_polars.to_cooler(part_path)
        part_paths.append(part_path)
        if print_progress:
            print(f"[INFO] Saved part {i} -> {part_path}")

    if not part_paths:
        raise RuntimeError("No cooler parts to merge")

    if print_progress:
        print(f"[INFO] Merging {len(part_paths)} parts into {res_path}")
    cooler.merge_coolers(output_uri=res_path, input_uris=part_paths, mergebuff=mergebuff)
    if print_progress:
        print(f"[INFO] Merged parts into: {res_path}")

    cleanup()
    return cooler.Cooler(res_path)