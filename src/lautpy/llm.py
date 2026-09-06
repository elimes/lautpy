#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenAI-compatible client factory.

Clients are built on demand from the environment and cached per
(service, key, base_url); the vendor list is open-ended — any service
exposing an OpenAI-compatible API works::

    export MOONSHOT_API_KEY=...          # optional: MOONSHOT_BASE_URL
    export ZHIPUAI_API_KEY=...           # optional: ZHIPUAI_BASE_URL

    from lautpy.llm import openai_client
    client = openai_client("moonshot")
"""

import os
from functools import lru_cache
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def _resolve(service: str, api_key: Optional[str], base_url: Optional[str]) -> tuple:
    upper = service.upper()
    api_key = api_key or os.getenv(f"{upper}_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API key for service '{service}'. Set {upper}_API_KEY or pass api_key=... "
            f"(keys are resolved from the environment, never hardcoded)."
        )
    base_url = base_url or os.getenv(f"{upper}_BASE_URL")
    return api_key, base_url


def openai_client(
    service: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Build (and cache) an ``openai.OpenAI`` client for any compatible service.

    The openai package is an optional dependency: ``pip install openai``.
    """
    if OpenAI is None:
        raise ImportError("openai is required: pip install openai")
    key, url = _resolve(service, api_key, base_url)
    return _cached_client(service, key, url)


@lru_cache(maxsize=32)
def _cached_client(service: str, api_key: str, base_url: Optional[str]) -> Any:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
