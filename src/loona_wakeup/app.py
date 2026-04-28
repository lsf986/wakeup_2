from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from loona_wakeup.adapters.live_udp_adapter import LiveUdpAdapter
from loona_wakeup.adapters.local_camera_mic_adapter import LocalCameraMicAdapter
from loona_wakeup.adapters.mock_adapter import MockAdapter
from loona_wakeup.config import load_config
from loona_wakeup.engine.decision_engine import WakeupDecisionEngine
from loona_wakeup.engine.utterance_gate import UtteranceGate
from loona_wakeup.models import MultimodalFrame, RunMode
from loona_wakeup.ui.main_window import MainWindow
from loona_wakeup.ui.theme import DARK_THEME


class LoonaWakeupApp:
    def __init__(self) -> None:
        self.qt_app = QApplication(sys.argv)
        self.qt_app.setStyleSheet(DARK_THEME)
        self.config = load_config()
        self.window = MainWindow(mode=self.config.runtime.mode.value.upper())
        self.engine = WakeupDecisionEngine(self.config.wakeup, self.config.weights)
        self.utterance_gate = UtteranceGate(self.config.wakeup)
        self.adapter = self._create_adapter()
        self.adapter.frame_ready.connect(self._on_frame)
        if hasattr(self.adapter, "status_changed"):
            self.adapter.status_changed.connect(self.window.set_runtime_status)
        if hasattr(self.adapter, "preview_ready"):
            self.adapter.preview_ready.connect(self.window.update_camera_frame)

    def run(self) -> int:
        self.window.show()
        self.adapter.start()
        return self.qt_app.exec()

    def _on_frame(self, frame: MultimodalFrame) -> None:
        if not self.window.running:
            return
        self.window.update_frame(frame)
        utterance_frames = self.utterance_gate.push(frame)
        if utterance_frames is None:
            return
        decision = self.engine.decide_utterance(utterance_frames)
        self.window.update_decision(decision)

    def _create_adapter(self) -> MockAdapter | LiveUdpAdapter | LocalCameraMicAdapter:
        if self.config.runtime.mode == RunMode.LOCAL:
            return LocalCameraMicAdapter(self.config.local_input)
        if self.config.runtime.mode == RunMode.LIVE:
            return LiveUdpAdapter(self.config.live_udp)
        return MockAdapter(interval_ms=self.config.runtime.decision_interval_ms)


def run() -> int:
    return LoonaWakeupApp().run()
