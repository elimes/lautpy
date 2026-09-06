#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal shared helpers — not part of the public API (hence the underscore)."""

import logging


def _build_logger():
    """loguru if available, else a minimally configured stdlib logger.

    Kept in one place so every module shares the same logging behavior.
    """
    try:
        from loguru import logger

        return logger
    except ImportError:
        lg = logging.getLogger("lautpy")
        if not lg.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)
        return lg


logger = _build_logger()
