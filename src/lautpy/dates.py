#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Date/time helpers (stdlib only).

"""

import datetime
import time
from typing import List, Optional, Union

DEFAULT_FMT = "%Y-%m-%d %H:%M:%S"


def date_difference(
    fmt: str = DEFAULT_FMT,
    start_date: Optional[Union[datetime.datetime, str, int]] = None,
    **kwargs,
) -> str:
    """Format `start_date` shifted by the given timedelta kwargs.

    kwargs accept any datetime.timedelta argument: days, seconds, minutes,
    hours, weeks. Positive values shift backwards ( subtraction semantics).

    Usage::

        date_difference(days=1)                                # yesterday
        date_difference('%Y%m%d', start_date=20210222, days=1)
    """
    start = start_date if start_date is not None else datetime.datetime.now()
    if isinstance(start, (str, int)):
        start = datetime.datetime.strptime(str(start), fmt)
    shifted = start - datetime.timedelta(**kwargs)
    return shifted.strftime(fmt)


def timestamp2str(timestamp: float, fmt: str = DEFAULT_FMT) -> str:
    """Second-level timestamp -> local time string."""
    return time.strftime(fmt, time.localtime(timestamp))


def str2timestamp(s: str, fmt: str = DEFAULT_FMT) -> float:
    """Local time string -> second-level timestamp."""
    return time.mktime(time.strptime(s, fmt))


def get_nday_list(n: int) -> List[str]:
    """The past `n` days as ISO date strings (today excluded, ascending)."""
    today = datetime.date.today()
    return [str(today - datetime.timedelta(days=i)) for i in range(n, 0, -1)]
