#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HTTP client infrastructure for third-party API wrappers.

Security model (the main lesson from meutils' apis/*): API keys are NEVER
hardcoded — they are resolved from environment variables at call time.
"""

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30  # seconds; meutils used no timeout at all


class MissingAPIKeyError(RuntimeError):
    """Raised when an API key is not configured in the environment."""


def get_api_key(service: str, env_var: Optional[str] = None) -> str:
    """Resolve an API key from the environment.

    Looks up `env_var` first, then `<SERVICE>_API_KEY` (e.g. NIUTRANS_API_KEY).
    """
    names = [env_var] if env_var else []
    default = f"{service.upper()}_API_KEY"
    if default not in names:
        names.append(default)
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise MissingAPIKeyError(
        f"API key for '{service}' not found. Set {' or '.join(names)} "
        f"in your environment. Never hardcode keys in source code."
    )


def http_session(retries: int = 2, backoff_factor: float = 0.5) -> requests.Session:
    """A requests session with retry on transient failures (429/5xx)."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def request(method: str, url: str, *, timeout: float = DEFAULT_TIMEOUT,
            retries: int = 2, session: Optional[requests.Session] = None,
            **kwargs) -> requests.Response:
    """request() with sane defaults: timeout (mandatory-ish) + retry + raise_for_status."""
    own_session = session is None
    s = session or http_session(retries=retries)
    try:
        response = s.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    finally:
        if own_session:
            s.close()
