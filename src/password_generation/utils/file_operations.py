import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from functools import partial
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm as std_tqdm

from password_generation.utils.logging import tqdm


def hash_string(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def hash_dict(x: Dict) -> str:
    return hash_string(json.dumps(x, sort_keys=True))


def bytes_to_gigabytes(x: int) -> float:
    return np.round(x / (1024**3), 2)


def get_file_size(file_path: str) -> float:
    assert os.path.isfile(file_path)
    return bytes_to_gigabytes(os.path.getsize(file_path))


def get_disk_usage(path: str) -> Tuple[str, float, float, float]:
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    total, used, free = shutil.disk_usage(path)
    return path, bytes_to_gigabytes(total), bytes_to_gigabytes(used), bytes_to_gigabytes(free)


def download_file(url: str, output_file: str, filetype: str = ".txt.gz") -> None:
    class _DownloadProgressBar(std_tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        file, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)

    if filetype == ".txt":
        os.rename(file, output_file)
    if filetype == ".txt.gz" or filetype == ".gz":
        batch_size: int = 1024**2  # 1MB batches
        # Approximate progress bar since unzipped file size is unknown.
        with tqdm(
            total=os.path.getsize(file) // batch_size * 1.5,
            desc="Unpacking binary file",
            unit="MB",
        ) as pbar:
            with open(output_file, "w", encoding="utf-8") as f_out:
                with gzip.open(file, "rb") as f_in:
                    while True:
                        file_content = f_in.read(batch_size).decode("utf-8", errors="ignore")
                        if not file_content:
                            break
                        f_out.write(file_content)
                        pbar.update(1)
