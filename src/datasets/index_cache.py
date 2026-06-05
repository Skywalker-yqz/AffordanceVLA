"""
Shared frame-index caching utilities for Stage II datasets.

Caches the expensive index-building results (file scanning, JSON parsing,
parquet reading, bbox validation) to a pickle file on disk so that
subsequent training runs skip the full pipeline (30+ min → seconds).

Each .pkl cache is accompanied by a human-readable .json metadata file
describing what parameters produced the cache, when, and how many samples.

Cache invalidation:
    - Parameters that affect the index are hashed into a fingerprint.
    - If ANY parameter changes, the fingerprint changes → cache miss → rebuild.
    - A CACHE_VERSION constant is included in the fingerprint; bump it to
      force global invalidation after code changes to the index format.
    - Users can also pass force_rescan=True to ignore all caches.

DDP safety:
    - All ranks build independently and may all write the same cache file.
    - Writes use tmp-file + atomic rename, so concurrent writes are safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def compute_fingerprint(*args) -> str:
    """Compute a 16-char hex fingerprint from arbitrary hashable arguments."""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}".encode())
    for a in args:
        h.update(repr(a).encode())
    return h.hexdigest()[:16]


def load_index_cache(
    cache_dir: Path,
    prefix: str,
    fingerprint: str,
) -> Optional[Any]:
    """Try to load a cached index from disk.

    Returns the cached object, or None if not found / corrupted.
    """
    cache_path = cache_dir / f".{prefix}_index_cache_{fingerprint}.pkl"
    if not cache_path.exists():
        return None
    try:
        t0 = time.time()
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        elapsed = time.time() - t0
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Loaded index cache: {cache_path.name} "
            f"({size_mb:.1f} MB, {elapsed:.1f}s)"
        )
        return data
    except Exception as e:
        logger.warning(f"Failed to load index cache {cache_path}: {e}")
        return None


def _remove_stale_caches(cache_dir: Path, prefix: str, keep_stem: str) -> None:
    """Delete old cache files (.pkl + .json) with the same prefix."""
    for suffix in ("*.pkl", "*.json"):
        pattern = f".{prefix}_index_cache_{suffix}"
        for old in cache_dir.glob(pattern):
            if old.stem == keep_stem or old.suffix == ".tmp":
                continue
            try:
                old.unlink()
                logger.info(f"Removed stale cache: {old.name}")
            except OSError:
                pass


def save_index_cache(
    cache_dir: Path,
    prefix: str,
    fingerprint: str,
    data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save index data to a pickle cache file (atomic write).

    Also writes a companion .json with human-readable metadata.
    Automatically removes stale cache files with the same prefix.
    """
    stem = f".{prefix}_index_cache_{fingerprint}"
    cache_path = cache_dir / f"{stem}.pkl"
    json_path = cache_dir / f"{stem}.json"
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        tmp_path.rename(cache_path)
        elapsed = time.time() - t0
        size_mb = cache_path.stat().st_size / (1024 * 1024)

        # Write companion JSON
        num_samples = len(data) if hasattr(data, "__len__") else "unknown"
        info: Dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "prefix": prefix,
            "fingerprint": fingerprint,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_samples": num_samples,
            "pkl_size_mb": round(size_mb, 2),
        }
        if metadata:
            info["params"] = metadata
        with open(json_path, "w") as jf:
            json.dump(info, jf, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved index cache: {cache_path.name} "
            f"({size_mb:.1f} MB, {elapsed:.1f}s)"
        )
        _remove_stale_caches(cache_dir, prefix, keep_stem=stem)
    except Exception as e:
        logger.warning(f"Failed to save index cache: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
