# @Author: elimes
"""可复现性：一键固定各随机源的种子。

PYTHONHASHSEED 注意：hash 随机化在解释器启动时已确定，运行期设置环境变量
对**当前进程无效**——需要跨进程复现时请在启动前导出该变量。
"""

import os
import random


def set_seed(seed: int = 42) -> int:
    """固定 random / numpy / torch（如已安装）的随机种子，返回种子值。

    Args:
        seed: 随机种子。

    Returns:
        int: 传入的种子（便于链式记录到实验元数据）。

    Example::

        seed = set_seed(2026)   # random + numpy + torch 全部固定
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except ImportError:
        pass

    return seed
