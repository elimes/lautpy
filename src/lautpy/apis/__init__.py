#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Third-party API wrappers with env-var-based key management.

Why only a few wrappers were ported from meutils.apis (325 files): most of
them are not standalone — they depend on the author's private infrastructure
(Feishu-sheets key pools, his own Aliyun OSS buckets, an one-api gateway) and
on meutils' internal schemas/llm/oss/caches stack. Those cannot be ported
meaningfully. New wrappers should follow the pattern in client.py: keys via
get_api_key(), all HTTP through request() (timeout + retry), no secrets in code.
"""

from lautpy.apis.client import (  # noqa: F401
    MissingAPIKeyError,
    get_api_key,
    http_session,
    request,
)
