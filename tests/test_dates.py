import time
from datetime import datetime

from lautpy.dates import date_difference, get_nday_list, str2timestamp, timestamp2str


def test_date_difference_days():
    start = datetime(2026, 9, 6, 12, 0, 0)
    assert date_difference(start_date=start, days=1) == "2026-09-05 12:00:00"


def test_date_difference_from_int():
    assert date_difference("%Y%m%d", start_date=20210222, days=1) == "20210221"


def test_timestamp_roundtrip():
    ts = time.mktime((2026, 9, 6, 12, 0, 0, 0, 0, -1))
    assert str2timestamp(timestamp2str(ts)) == ts


def test_get_nday_list():
    days = get_nday_list(3)
    assert len(days) == 3 and days == sorted(days)
