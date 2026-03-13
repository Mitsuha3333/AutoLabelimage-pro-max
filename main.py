from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch  # noqa: F401  # 必须在 PyQt5 之前导入，否则 Windows 上 DLL 加载冲突
except ImportError:
    pass

from PyQt5 import QtWidgets

from app.widgets.main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AutoLabelProMax")
    config_path = Path(__file__).resolve().parent / "config.yaml"
    window = MainWindow(config_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
