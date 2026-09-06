# @Author: elimes
"""统一工具定义：把普通函数变成 Agent 可调用的工具。

只需写一个带类型标注和中文 docstring 的普通函数，@tool 装饰后：
- 原生引擎直接使用（自动生成 OpenAI function-calling 的 JSON Schema）
- LangChain 引擎通过 to_langchain() 转换，无重复定义
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

# Python 类型 → OpenAI JSON Schema 类型
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_hints(func: Callable) -> dict[str, Any]:
    """从函数签名推导 JSON Schema（required = 无默认值的参数）。"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__module__") else {}
    properties: dict[str, Any] = {}
    required = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        hint = hints.get(name, str)
        json_type = _TYPE_MAP.get(hint)
        origin = getattr(hint, "__origin__", None)  # List[str] 等泛型的容器类型
        if json_type is None and origin is not None:
            json_type = _TYPE_MAP.get(origin, "string")
        properties[name] = {"type": json_type or "string"}
        if param.default is param.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class Tool:
    """Agent 工具对象：包装普通函数，供原生引擎与 LangChain 引擎共用。

    通常不直接实例化，用 ``@tool`` 装饰器获得。
    """

    def __init__(self, func: Callable):
        if not callable(func):
            raise TypeError("tool() expects a plain function")
        self.func = func
        self.name = func.__name__
        self.description = (inspect.getdoc(func) or "").strip()
        if not self.description:
            raise ValueError(f"tool {self.name}() needs a docstring as its description")
        self.parameters = _schema_from_hints(func)
        functools.update_wrapper(self, func)

    def to_openai_spec(self) -> dict[str, Any]:
        """生成 OpenAI function-calling 的 tools 元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_langchain(self):
        """转换为 langchain 工具（StructuredTool），需安装 langchain-core。"""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError('langchain is required: pip install "lautpy[agent]"') from e
        return StructuredTool.from_function(self.func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name}, params={list(self.parameters.get('properties', []))})"


def tool(func: Callable | None = None) -> Tool:
    """把普通函数变成 Agent 工具；函数 docstring 即工具描述（必填）。

    Example::

        @tool
        def get_weather(city: str) -> str:
            '''查询指定城市的实时天气。'''
            ...
    """
    if func is None:  # 允许 @tool 不带括号的裸用法之外的 @tool() 写法
        raise TypeError("use @tool directly, without parentheses")
    return Tool(func)
