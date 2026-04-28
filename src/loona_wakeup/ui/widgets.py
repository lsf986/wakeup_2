from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout, QWidget


def panel() -> QFrame:
    frame = QFrame()
    frame.setObjectName("Panel")
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    return frame


class MetricRow(QWidget):
    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name_label = QLabel(name)
        self.name_label.setObjectName("MetricName")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.addWidget(self.name_label, 0, 0)
        layout.addWidget(self.value_label, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class ScoreBar(QWidget):
    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel(name)
        self.label.setObjectName("MetricName")
        self.value = QLabel("0.00")
        self.value.setObjectName("MetricValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.value, 0, 1)
        layout.addWidget(self.bar, 1, 0, 1, 2)

    def set_score(self, score: float) -> None:
        score = max(0.0, min(1.0, score))
        self.value.setText(f"{score:.2f}")
        self.bar.setValue(int(score * 100))


def titled_panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = panel()
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    title_label = QLabel(title)
    title_label.setObjectName("Subtle")
    layout.addWidget(title_label)
    return frame, layout
