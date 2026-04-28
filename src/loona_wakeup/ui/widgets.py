from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


def panel() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Panel")
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    return frame


def titled_panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = panel()
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    title_label = QLabel(title)
    title_label.setObjectName("Subtle")
    layout.addWidget(title_label)
    return frame, layout
