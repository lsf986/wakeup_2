from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest_image: QImage | None = None
        self.label = QLabel("Camera waiting for local input")
        self.label.setObjectName("Subtle")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 420)
        self.label.setStyleSheet("background: #050608; border: 1px solid #242A33; border-radius: 6px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

    def set_frame(self, image: QImage) -> None:
        self._latest_image = image
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self._latest_image is not None:
            self.set_frame(self._latest_image)
        super().resizeEvent(event)

    def stop(self) -> None:
        return
