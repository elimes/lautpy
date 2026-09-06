#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Third-party API wrappers with env-var-based key management.

Conventions for every wrapper in this package:

- keys via ``get_api_key()`` (environment only, never hardcoded)
- all HTTP through ``request()`` (timeout + retry)
- payload building kept pure so it can be tested without network
"""

from lautpy.apis.client import (  # noqa: F401
    MissingAPIKeyError,
    get_api_key,
    http_session,
    request,
)
