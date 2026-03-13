from __future__ import annotations

from typing import Iterable


DEFAULT_PROMPT_TEMPLATE = """你是一个严格的视觉标注器。

任务：
只检测图中所有清晰可见的以下目标：
{class_bullets}

只返回 JSON 数组，不要返回任何解释。

格式必须严格为：
[
  {"label":"类别名","bbox":[x1,y1,x2,y2]}
]

规则：
- label 必须严格来自给定类别列表
- bbox 使用 0~1000 的相对坐标
- 不要使用像素坐标
- 坐标原点是图像左上角
- x1 < x2
- y1 < y2
- bbox 必须贴合目标边界
- 不要重复输出
- 不确定就不要输出
- 没有目标时返回 []

最终答案只能是 JSON 数组。
"""


def build_class_bullets(class_names: Iterable[str]) -> str:
    names = [name.strip() for name in class_names if name.strip()]
    if not names:
        return "- 暂无类别"
    return "\n".join(f"- {name}" for name in names)


def build_prompt(template: str, class_names: Iterable[str]) -> str:
    class_bullets = build_class_bullets(class_names)
    if "{class_bullets}" in template:
        return template.replace("{class_bullets}", class_bullets)
    return f"{template.rstrip()}\n\n{class_bullets}\n"
