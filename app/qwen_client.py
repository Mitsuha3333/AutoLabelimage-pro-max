from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

import requests

from .config import QwenConfig
from .models import BoxAnnotation, make_id
from .utils import clamp, image_to_data_url, rel1000_to_xyxy


class QwenClientError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def detect(self, image_path: str | Path, class_names: Iterable[str], prompt: str) -> List[BoxAnnotation]:
        classes = [name.strip() for name in class_names if name.strip()]
        if not classes:
            return []

        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
        }

        try:
            response = self.session.post(
                self._chat_url(),
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
        except requests.RequestException as exc:
            raise QwenClientError(f"Qwen 接口不可用: {exc}") from exc
        except ValueError as exc:
            raise QwenClientError(f"Qwen 返回不是合法 JSON: {exc}") from exc

        content = self._extract_content(response_data)
        items = self._extract_json_array(content)
        return self._to_annotations(items, classes, Path(image_path))

    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}{self.config.chat_completions_path}"

    def _extract_content(self, response_data: dict[str, Any]) -> str:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenClientError("Qwen 返回结构不符合 OpenAI 兼容格式") from exc

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if text:
                    parts.append(str(text))
            return "\n".join(parts)

        return str(content)

    def _extract_json_array(self, content: str) -> list[dict[str, Any]]:
        stripped = content.strip()
        candidates = [stripped]
        candidates.extend(self._iter_json_array_candidates(stripped))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]

        raise QwenClientError("无法从模型返回内容中提取 JSON 数组")

    def _iter_json_array_candidates(self, text: str) -> list[str]:
        results: list[str] = []
        for start in range(len(text)):
            if text[start] != "[":
                continue
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(text)):
                char = text[end]
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        results.append(text[start : end + 1])
                        break
        return results

    def _to_annotations(
        self,
        items: list[dict[str, Any]],
        class_names: list[str],
        image_path: Path,
    ) -> List[BoxAnnotation]:
        from PIL import Image

        label_to_id = {name: index for index, name in enumerate(class_names)}
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        result: List[BoxAnnotation] = []
        for item in items:
            label = str(item.get("label", "")).strip()
            if label not in label_to_id:
                continue

            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue

            x1 = clamp(x1, 0.0, 1000.0)
            y1 = clamp(y1, 0.0, 1000.0)
            x2 = clamp(x2, 0.0, 1000.0)
            y2 = clamp(y2, 0.0, 1000.0)
            if x1 >= x2 or y1 >= y2:
                continue

            rel_bbox = (x1, y1, x2, y2)
            result.append(
                BoxAnnotation(
                    id=make_id("box"),
                    label=label,
                    class_id=label_to_id[label],
                    source="qwen",
                    bbox_xyxy=rel1000_to_xyxy(rel_bbox, image_width, image_height),
                    bbox_rel_1000=rel_bbox,
                )
            )
        return result
