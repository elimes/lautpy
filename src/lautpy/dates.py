# @Author: elimes
"""日期与时间戳转换工具（纯标准库，无第三方依赖）。

时间语义统一为**本地时间**：timestamp2str/str2timestamp 互为逆运算，
均使用本机时区，不做跨时区换算。
"""

import datetime
import time

DEFAULT_FMT = "%Y-%m-%d %H:%M:%S"


def date_difference(
    fmt: str = DEFAULT_FMT,
    start_date: datetime.datetime | str | int | None = None,
    **kwargs,
) -> str:
    """计算 start_date 平移指定时间量后的日期字符串。

    Args:
        fmt: 输入/输出共用的日期格式（strftime 风格）。
        start_date: 起点时间。datetime 对象原样使用；str/int 按 fmt 解析；
            缺省为当前时间。
        **kwargs: 任意 datetime.timedelta 参数（days/seconds/minutes/
            hours/weeks），按"减法"语义平移，正值表示向过去偏移。

    Returns:
        str: 平移后按 fmt 格式化的日期字符串。

    Example::

        date_difference(days=1)                                  # 昨天
        date_difference('%Y%m%d', start_date=20210222, days=1)   # '20210221'
    """
    start = start_date if start_date is not None else datetime.datetime.now()
    if isinstance(start, (str, int)):
        start = datetime.datetime.strptime(str(start), fmt)
    shifted = start - datetime.timedelta(**kwargs)
    return shifted.strftime(fmt)


def timestamp2str(timestamp: float, fmt: str = DEFAULT_FMT) -> str:
    """秒级时间戳 → 本地时间字符串。

    Args:
        timestamp: 秒级 Unix 时间戳。
        fmt: 输出日期格式。

    Example::

        timestamp2str(1787000000.0)   # '2026-08-14 11:33:20'
    """
    return time.strftime(fmt, time.localtime(timestamp))


def str2timestamp(s: str, fmt: str = DEFAULT_FMT) -> float:
    """本地时间字符串 → 秒级时间戳（timestamp2str 的逆运算）。

    Args:
        s: 时间字符串。
        fmt: 解析格式，须与 s 一致。
    """
    return time.mktime(time.strptime(s, fmt))


def get_nday_list(n: int) -> list[str]:
    """获取过去 n 天的日期列表（不含今天，升序，ISO 格式 'YYYY-MM-DD'）。

    Args:
        n: 天数，须为正整数。
    """
    today = datetime.date.today()
    return [str(today - datetime.timedelta(days=i)) for i in range(n, 0, -1)]
