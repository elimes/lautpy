#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : X.
# @File         : __init__.py
# @Time         : 2020/11/12 10:54 上午
# @Author       : liufeng
# @Email        : elimes@qq.com
# @Software     : PyCharm
# @Description  :

from pathlib import Path
import os
from lautpy.pipe import *

__version__ = Path(get_resolve_path('./data/VERSION', __file__)).read_text()
