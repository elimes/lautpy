# @Author: elimes
"""机器学习辅助层：可复现性、指标补集、分箱 WOE/IV、模型对比、数据划分、模型存取。

定位是"胶水层"：只补 sklearn 没有直接给的日常刚需，不重造标准算法。
numpy 为可选依赖（缺 numpy 时 import 本包不报错，调用时给出指引）。
"""

from lautpy.ml.seed import set_seed

__all__ = ["set_seed"]
