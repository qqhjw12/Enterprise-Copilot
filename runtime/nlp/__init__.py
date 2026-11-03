"""轻量自然语言处理：不依赖模型的可确定性解析。

放这里的都是"纯函数、零依赖、可单测"的东西 —— 在离线降级模式下，
它们是系统唯一的理解能力，所以必须比正则更结构化一点。
"""
from runtime.nlp.dates import DateParse, merge_user_texts, parse_date_range

__all__ = ["DateParse", "merge_user_texts", "parse_date_range"]
