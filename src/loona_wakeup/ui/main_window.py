from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from loona_wakeup.models import WakeupDecision
from loona_wakeup.ui.camera_preview import CameraPreview
from loona_wakeup.ui.widgets import panel, titled_panel

WAKEUP_HOLD_MS = 1000


class MainWindow(QMainWindow):
    def __init__(self, mode: str = "MOCK") -> None:
        super().__init__()
        self.setWindowTitle("Loona Wakeup Monitor")
        self.resize(900, 640)
        self._running = True
        self._mode = mode
        self._source_status = mode
        self._last_wakeup_feedback_ms = -WAKEUP_HOLD_MS
        self._wakeup_hold_timer = QTimer(self)
        self._wakeup_hold_timer.setSingleShot(True)
        self._wakeup_hold_timer.timeout.connect(self._set_idle_state)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        root.addLayout(self._build_header())
        root.addWidget(self._build_state_panel())
        root.addWidget(self._build_camera_panel(), 1)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        title = QLabel("Loona Wakeup Monitor")
        title.setObjectName("Title")
        self.mode_label = QLabel(f"{self._source_status}  ●  RUNNING")
        self.mode_label.setObjectName("Subtle")
        self.toggle_button = QPushButton("Pause")
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._toggle_running)

        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.toggle_button)
        return layout

    def _build_state_panel(self) -> QFrame:
        frame = panel()
        layout = QGridLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setHorizontalSpacing(20)

        self.state_label = QLabel("IDLE")
        self.state_label.setObjectName("StateIdle")

        layout.addWidget(self.state_label, 0, 0)
        layout.setColumnStretch(0, 1)
        return frame

    def _build_camera_panel(self) -> QFrame:
        frame, layout = titled_panel("Camera")
        self.camera_preview = CameraPreview()
        layout.addWidget(self.camera_preview, 1)
        return frame

    def _toggle_running(self) -> None:
        self._running = not self.toggle_button.isChecked()
        self.toggle_button.setText("Resume" if not self._running else "Pause")
        state = "PAUSED" if not self._running else "RUNNING"
        self.mode_label.setText(f"{self._source_status}  ●  {state}")

    def set_runtime_status(self, status: str) -> None:
        self._source_status = status
        state = "RUNNING" if self._running else "PAUSED"
        self.mode_label.setText(f"{self._source_status}  ●  {state}")

    def update_camera_frame(self, image) -> None:  # noqa: ANN001
        self.camera_preview.set_frame(image)

    @property
    def running(self) -> bool:
        return self._running

    def update_decision(self, decision: WakeupDecision) -> None:
        if not self.running:
            return

        if decision.wakeup:
            self._set_wakeup_state()
            self._play_wakeup_feedback(decision.timestamp_ms)
            self._wakeup_hold_timer.start(WAKEUP_HOLD_MS)
        elif not self._wakeup_hold_timer.isActive():
            self._set_idle_state()

    def _set_wakeup_state(self) -> None:
        self.state_label.setText("WAKEUP")
        self.state_label.setObjectName("StateWakeup")
        self._refresh_state_style()

    def _play_wakeup_feedback(self, timestamp_ms: int) -> None:
        if timestamp_ms - self._last_wakeup_feedback_ms < WAKEUP_HOLD_MS:
            return
        self._last_wakeup_feedback_ms = timestamp_ms
        QApplication.beep()
        self.camera_preview.flash_wakeup_border()

    def _set_idle_state(self) -> None:
        self.state_label.setText("IDLE")
        self.state_label.setObjectName("StateIdle")
        self._refresh_state_style()

    def _refresh_state_style(self) -> None:
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

