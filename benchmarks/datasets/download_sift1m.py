"""
Dataset loader and parser for SIFT benchmarks (siftsmall / SIFT1M).

SIFT datasets (Texmex corpus) use .fvecs and .ivecs binary formats:
- .fvecs: Each vector begins with a 4-byte int32 specifying dimension d,
          followed by d 32-bit floats. Total record size = 4 + 4*d bytes.
- .ivecs: Each vector begins with a 4-byte int32 specifying dimension d,
          followed by d 32-bit integers. Total record size = 4 + 4*d bytes.

This module downloads siftsmall (10k 128-dim base vectors, 100 queries, 100 ground truth neighbors),
parses .fvecs / .ivecs files, and caches them as NumPy .npy files for fast loading.
"""

import os
import tarfile
import urllib.request
import numpy as np
from typing import Optional, Tuple

SIFTSMALL_URLS = [
    "https://github.com/TileDB-Inc/TileDB-Vector-Search/releases/download/0.0.1/siftsmall.tgz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
]


def read_fvecs(filename: str) -> np.ndarray:
    """
    Read a .fvecs file into a NumPy float32 array.

    Format:
        Each vector is stored as:
        [dim: int32 (4 bytes)] [v_0, v_1, ..., v_{dim-1}: float32 (4*dim bytes)]

    Args:
        filename: Path to .fvecs file.

    Returns:
        np.ndarray of shape (num_vectors, dim) and dtype np.float32.
    """
    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes or len(dim_bytes) < 4:
            return np.empty((0, 0), dtype=np.float32)
        dim = int(np.frombuffer(dim_bytes, dtype="<i4")[0])

    dt = np.dtype([("dim", "<i4"), ("vector", "<f4", (dim,))])
    data = np.fromfile(filename, dtype=dt)
    return data["vector"].copy()


def read_ivecs(filename: str) -> np.ndarray:
    """
    Read an .ivecs file into a NumPy int32 array.

    Format:
        Each vector is stored as:
        [dim: int32 (4 bytes)] [id_0, id_1, ..., id_{dim-1}: int32 (4*dim bytes)]

    Args:
        filename: Path to .ivecs file.

    Returns:
        np.ndarray of shape (num_vectors, dim) and dtype np.int32.
    """
    with open(filename, "rb") as f:
        dim_bytes = f.read(4)
        if not dim_bytes or len(dim_bytes) < 4:
            return np.empty((0, 0), dtype=np.int32)
        dim = int(np.frombuffer(dim_bytes, dtype="<i4")[0])

    dt = np.dtype([("dim", "<i4"), ("vector", "<i4", (dim,))])
    data = np.fromfile(filename, dtype=dt)
    return data["vector"].copy()


def write_fvecs(filename: str, vectors: np.ndarray) -> None:
    """
    Write a 2D float32 array to .fvecs format.
    Used for unit testing and data export.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    num_vectors, dim = vectors.shape
    dt = np.dtype([("dim", "<i4"), ("vector", "<f4", (dim,))])
    structured = np.empty(num_vectors, dtype=dt)
    structured["dim"] = dim
    structured["vector"] = vectors
    structured.tofile(filename)


def write_ivecs(filename: str, vectors: np.ndarray) -> None:
    """
    Write a 2D int32 array to .ivecs format.
    Used for unit testing and data export.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.int32)
    num_vectors, dim = vectors.shape
    dt = np.dtype([("dim", "<i4"), ("vector", "<i4", (dim,))])
    structured = np.empty(num_vectors, dtype=dt)
    structured["dim"] = dim
    structured["vector"] = vectors
    structured.tofile(filename)


def download_and_extract_siftsmall(target_dir: str) -> None:
    """
    Download and extract siftsmall archive into target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, "siftsmall.tar.gz")
    tgz_path = os.path.join(target_dir, "siftsmall.tgz")

    # Check if archive already exists
    chosen_archive = None
    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 1000000:
        chosen_archive = archive_path
    elif os.path.exists(tgz_path) and os.path.getsize(tgz_path) > 1000000:
        chosen_archive = tgz_path

    if chosen_archive is None:
        downloaded = False
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        for url in SIFTSMALL_URLS:
            try:
                print(f"Downloading siftsmall from: {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as out:
                    while chunk := resp.read(64 * 1024):
                        out.write(chunk)
                print(f"Downloaded successfully: {os.path.getsize(archive_path)} bytes")
                downloaded = True
                chosen_archive = archive_path
                break
            except Exception as e:
                print(f"Download failed from {url}: {e}")

        if not downloaded or chosen_archive is None:
            raise RuntimeError("Failed to download siftsmall from all available sources.")

    print(f"Extracting {chosen_archive} to {target_dir}...")
    with tarfile.open(chosen_archive, "r:*") as tar:
        tar.extractall(path=target_dir)
    print("Extraction complete.")


def load_siftsmall(
    target_dir: Optional[str] = None, force_reparse: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SIFT small dataset (base, query, ground truth).
    Caches parsed arrays as .npy files in the dataset directory for fast reloading.

    Returns:
        Tuple of (base_vectors, query_vectors, groundtruth_ids):
        - base_vectors: np.ndarray of shape (10000, 128), float32
        - query_vectors: np.ndarray of shape (100, 128), float32
        - groundtruth_ids: np.ndarray of shape (100, 100), int32
    """
    if target_dir is None:
        target_dir = os.path.dirname(os.path.abspath(__file__))

    base_npy = os.path.join(target_dir, "siftsmall_base.npy")
    query_npy = os.path.join(target_dir, "siftsmall_query.npy")
    gt_npy = os.path.join(target_dir, "siftsmall_groundtruth.npy")

    # Fast path: load cached .npy
    if (
        not force_reparse
        and os.path.exists(base_npy)
        and os.path.exists(query_npy)
        and os.path.exists(gt_npy)
    ):
        print(f"Loading cached SIFT dataset from {target_dir}...")
        base = np.load(base_npy)
        query = np.load(query_npy)
        gt = np.load(gt_npy)
        return base, query, gt

    # Look for raw files in target_dir or target_dir/siftsmall
    raw_candidates = [
        target_dir,
        os.path.join(target_dir, "siftsmall"),
    ]

    base_fvecs = None
    query_fvecs = None
    gt_ivecs = None

    for candidate in raw_candidates:
        b = os.path.join(candidate, "siftsmall_base.fvecs")
        q = os.path.join(candidate, "siftsmall_query.fvecs")
        g = os.path.join(candidate, "siftsmall_groundtruth.ivecs")
        if os.path.exists(b) and os.path.exists(q) and os.path.exists(g):
            base_fvecs = b
            query_fvecs = q
            gt_ivecs = g
            break

    if base_fvecs is None:
        print("Raw dataset files not found. Downloading...")
        download_and_extract_siftsmall(target_dir)
        # Re-check paths after extract
        for candidate in raw_candidates:
            b = os.path.join(candidate, "siftsmall_base.fvecs")
            q = os.path.join(candidate, "siftsmall_query.fvecs")
            g = os.path.join(candidate, "siftsmall_groundtruth.ivecs")
            if os.path.exists(b) and os.path.exists(q) and os.path.exists(g):
                base_fvecs = b
                query_fvecs = q
                gt_ivecs = g
                break

    if base_fvecs is None or query_fvecs is None or gt_ivecs is None:
        raise FileNotFoundError("Failed to locate extracted siftsmall .fvecs/.ivecs files.")

    print(f"Parsing raw fvecs/ivecs files from {os.path.dirname(base_fvecs)}...")
    base = read_fvecs(base_fvecs)
    query = read_fvecs(query_fvecs)
    gt = read_ivecs(gt_ivecs)

    print(f"Parsed base vectors: shape={base.shape}, dtype={base.dtype}")
    print(f"Parsed query vectors: shape={query.shape}, dtype={query.dtype}")
    print(f"Parsed ground truth: shape={gt.shape}, dtype={gt.dtype}")

    # Cache as .npy
    print(f"Saving cached .npy files to {target_dir}...")
    np.save(base_npy, base)
    np.save(query_npy, query)
    np.save(gt_npy, gt)

    return base, query, gt


if __name__ == "__main__":
    # Self-test: parse or download siftsmall
    base, query, gt = load_siftsmall()
    print("\nDataset successfully loaded and validated:")
    print(f"  Base vectors:   {base.shape} (expected 10000, 128)")
    print(f"  Query vectors:  {query.shape} (expected 100, 128)")
    print(f"  Ground truth:   {gt.shape} (expected 100, 100)")
    assert base.shape == (10000, 128)
    assert query.shape == (100, 128)
    assert gt.shape == (100, 100)
    print("Dataset validation passed!")
