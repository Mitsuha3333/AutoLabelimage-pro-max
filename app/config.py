from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .prompt_builder import DEFAULT_PROMPT_TEMPLATE


@dataclass
class QwenConfig:
    base_url: str = "http://127.0.0.1:8887"
    chat_completions_path: str = "/v1/chat/completions"
    timeout: int = 60
    model: str = "Qwen3VL-8B-Instruct"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE


@dataclass
class SamConfig:
    model_type: str = "vit_b"
    checkpoint: str = "./models/sam_vit_b.pth"
    device: str = "cuda"
    expand_ratio: float = 0.02
    min_box_area: int = 16
    min_area_ratio: float = 0.6
    max_area_ratio: float = 1.15
    max_center_shift_ratio: float = 0.15


@dataclass
class ProjectConfig:
    auto_save: bool = True
    save_intermediate_json: bool = True
    classes_file: str = "classes.txt"


@dataclass
class AppConfig:
    qwen: QwenConfig
    sam: SamConfig
    project: ProjectConfig


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig(qwen=QwenConfig(), sam=SamConfig(), project=ProjectConfig())

    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        qwen=_load_qwen_config(raw_data.get("qwen", {})),
        sam=_load_sam_config(raw_data.get("sam", {})),
        project=_load_project_config(raw_data.get("project", {})),
    )


def _load_qwen_config(data: dict[str, Any]) -> QwenConfig:
    return QwenConfig(
        base_url=data.get("base_url", QwenConfig.base_url),
        chat_completions_path=data.get("chat_completions_path", QwenConfig.chat_completions_path),
        timeout=int(data.get("timeout", QwenConfig.timeout)),
        model=data.get("model", QwenConfig.model),
        prompt_template=data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
    )


def _load_sam_config(data: dict[str, Any]) -> SamConfig:
    return SamConfig(
        model_type=data.get("model_type", SamConfig.model_type),
        checkpoint=data.get("checkpoint", SamConfig.checkpoint),
        device=data.get("device", SamConfig.device),
        expand_ratio=float(data.get("expand_ratio", SamConfig.expand_ratio)),
        min_box_area=int(data.get("min_box_area", SamConfig.min_box_area)),
        min_area_ratio=float(data.get("min_area_ratio", SamConfig.min_area_ratio)),
        max_area_ratio=float(data.get("max_area_ratio", SamConfig.max_area_ratio)),
        max_center_shift_ratio=float(data.get("max_center_shift_ratio", SamConfig.max_center_shift_ratio)),
    )


def _load_project_config(data: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        auto_save=bool(data.get("auto_save", ProjectConfig.auto_save)),
        save_intermediate_json=bool(data.get("save_intermediate_json", ProjectConfig.save_intermediate_json)),
        classes_file=data.get("classes_file", ProjectConfig.classes_file),
    )
