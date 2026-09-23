from __future__ import annotations

import os
from pathlib import Path


def default_hf_cache_dir() -> str:
    """Return a stable default Hugging Face cache path.

    Windows users usually want this on D: to avoid filling C:.
    On macOS/Linux, fall back to a folder in the user's home directory.
    """
    if os.name == "nt":
        return r"D:\huggingface_cache"
    return str(Path.home() / "huggingface_cache")


def configure_hf_cache(cache_dir: str | None = None) -> str:
    """Force Hugging Face/Transformers to reuse one cache directory.

    This function is safe to call repeatedly. It sets the environment variables
    used by huggingface_hub, transformers and datasets, and creates the folders.
    The actual model loading code also passes cache_dir explicitly, so this works
    even when transformers was already imported.
    """
    cache_root = Path((cache_dir or default_hf_cache_dir()).strip()).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    for path in (hub_cache, datasets_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    return str(cache_root)
