from __future__ import annotations

from typing import Iterable, List

from .models import ClassItem, make_id
from .utils import pick_color, unique_names


class ClassManager:
    def __init__(self, classes: Iterable[ClassItem] | None = None) -> None:
        self._classes: List[ClassItem] = list(classes or [])

    @property
    def classes(self) -> List[ClassItem]:
        return [ClassItem(**vars(item)) for item in self._classes]

    def set_classes_from_names(self, names: Iterable[str]) -> List[ClassItem]:
        clean_names = unique_names(names)
        self._classes = [
            ClassItem(id=make_id("class"), name=name, color=pick_color(index))
            for index, name in enumerate(clean_names)
        ]
        return self.classes

    def add_class(self, name: str) -> List[ClassItem]:
        clean_name = name.strip()
        if not clean_name or clean_name in self.class_names():
            return self.classes
        self._classes.append(
            ClassItem(id=make_id("class"), name=clean_name, color=pick_color(len(self._classes)))
        )
        return self.classes

    def remove_class(self, index: int) -> List[ClassItem]:
        if 0 <= index < len(self._classes):
            self._classes.pop(index)
            self._reassign_colors()
        return self.classes

    def rename_class(self, index: int, new_name: str) -> List[ClassItem]:
        if not 0 <= index < len(self._classes):
            return self.classes
        clean_name = new_name.strip()
        if not clean_name:
            return self.classes
        if clean_name in self.class_names() and self._classes[index].name != clean_name:
            return self.classes
        self._classes[index].name = clean_name
        return self.classes

    def move_up(self, index: int) -> List[ClassItem]:
        if 1 <= index < len(self._classes):
            self._classes[index - 1], self._classes[index] = self._classes[index], self._classes[index - 1]
            self._reassign_colors()
        return self.classes

    def move_down(self, index: int) -> List[ClassItem]:
        if 0 <= index < len(self._classes) - 1:
            self._classes[index + 1], self._classes[index] = self._classes[index], self._classes[index + 1]
            self._reassign_colors()
        return self.classes

    def class_names(self) -> List[str]:
        return [item.name for item in self._classes]

    def index_of(self, label: str) -> int:
        try:
            return self.class_names().index(label)
        except ValueError:
            return -1

    def _reassign_colors(self) -> None:
        for index, item in enumerate(self._classes):
            item.color = pick_color(index)
