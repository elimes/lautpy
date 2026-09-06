#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Path & config-file helpers.

"""

import json
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def get_resolve_path(path: Union[str, Path], file: Union[str, Path]) -> Path:
    """Resolve `path` relative to the directory containing `file` (usually __file__)."""
    return (Path(file).parent / Path(path)).resolve()


def _load_structured(text_or_bytes) -> dict:
    if yaml is not None:
        return yaml.safe_load(text_or_bytes)
    raise ImportError("pyyaml is required for yaml support: pip install pyyaml")


def file2json(path: Union[str, Path]) -> dict:
    """Load a .json / .yml / .yaml file into a dict; returns {} for invalid paths."""
    p = Path(path)
    if not p.is_file():
        return {}
    if p.name.endswith(".json"):
        return json.loads(p.read_bytes())
    if p.name.endswith((".yml", ".yaml")):
        return _load_structured(p.read_bytes()) or {}
    return {}


def path2list(path: Union[str, Path], pattern: str = "*") -> List[Path]:
    """Expand a file or directory (glob `pattern`) into a list of Paths."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return list(p.glob(pattern))
    return []


def get_config(config_init: Optional[Union[str, dict]]) -> dict:
    """Accept a dict, a path to a json/yaml file, or None; always return a dict."""
    if isinstance(config_init, str) and Path(config_init).is_file():
        return file2json(config_init)
    if isinstance(config_init, dict):
        return config_init
    return {}


def pkl_dump(obj: Any, file: Union[str, Path]) -> Path:
    """Pickle `obj` to `file` (binary); returns the path."""
    p = Path(file)
    p.write_bytes(pickle.dumps(obj, protocol=4))
    return p


def pkl_load(file: Union[str, Path]) -> Any:
    """Load an object pickled by pkl_dump."""
    return pickle.loads(Path(file).read_bytes())
