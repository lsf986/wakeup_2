from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest_image: QImage | None = None
        self._flash_remaining = 0
        self._flash_on = False
        self._base_style = "background: #050608; border: 1px solid #242A33; border-radius: 6px;"
        self._flash_style = "background: #050608; border: 3px solid #3FB950; border-radius: 6px;"
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(130)
        self._flash_timer.timeout.connect(self._advance_flash)
        self.label = QLabel("Camera waiting for local input")
        self.label.setObjectName("Subtle")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 420)
        self.label.setStyleSheet(self._base_style)

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

    def flash_wakeup_border(self) -> None:
        self._flash_remaining = 6
        self._flash_on = False
        self._advance_flash()
        self._flash_timer.start()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self._latest_image is not None:
            self.set_frame(self._latest_image)
        super().resizeEvent(event)

    def _advance_flash(self) -> None:
        if self._flash_remaining <= 0:
            self._flash_timer.stop()
            self._flash_on = False
            self.label.setStyleSheet(self._base_style)
            return

        self._flash_on = not self._flash_on
        self.label.setStyleSheet(self._flash_style if self._flash_on else self._base_style)
        self._flash_remaining -= 1

