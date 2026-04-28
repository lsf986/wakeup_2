from __future__ import annotations

import random
import time
from dataclasses import replace

from PySide6.QtCore import QObject, QTimer, Signal

from loona_wakeup.models import MultimodalFrame


class MockAdapter(QObject):
    frame_ready = Signal(MultimodalFrame)

    def __init__(self, interval_ms: int = 100, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._emit_frame)
        self._tick = 0
        self._scenario_index = 0
        self._scenarios = [
            self._idle_frame,
            self._valid_wakeup_frame,
            self._background_audio_frame,
            self._side_talk_frame,
            self._short_sentence_frame,
        ]

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _emit_frame(self) -> None:
        self._tick += 1
        if self._tick % 45 == 0:
            self._scenario_index = (self._scenario_index + 1) % len(self._scenarios)
        frame = self._scenarios[self._scenario_index]()
        self.frame_ready.emit(frame)

    def _base_frame(self, scene_type: str) -> MultimodalFrame:
        return MultimodalFrame(
            timestamp_ms=int(time.time() * 1000),
            user_id="user_01",
            scene_type=scene_type,
        )

    def _jitter(self, value: float, amount: float = 0.05) -> float:
        return max(0.0, min(1.0, value + random.uniform(-amount, amount)))

    def _idle_frame(self) -> MultimodalFrame:
        return replace(
            self._base_frame("idle"),
            has_voice=False,
            voice_energy=self._jitter(0.05),
            speech_like_score=self._jitter(0.06),
            sound_direction_deg=random.uniform(-70, 70),
            sound_distance_m=random.uniform(1.8, 3.2),
            face_visible=True,
            head_yaw_deg=random.uniform(-35, 35),
            gaze_to_loona_score=self._jitter(0.12),
            lip_movement_score=self._jitter(0.03),
            is_attention_target=False,
            background_audio_score=self._jitter(0.08),
        )

    def _valid_wakeup_frame(self) -> MultimodalFrame:
        return replace(
            self._base_frame("single_user_facing"),
            has_voice=True,
            voice_energy=self._jitter(0.86),
            speech_like_score=self._jitter(0.90),
            sound_direction_deg=random.uniform(-8, 8),
            sound_distance_m=random.uniform(0.55, 1.05),
            face_visible=True,
            head_yaw_deg=random.uniform(-8, 8),
            gaze_to_loona_score=self._jitter(0.78),
            lip_movement_score=self._jitter(0.76),
            is_attention_target=True,
            background_audio_score=self._jitter(0.05),
        )

    def _background_audio_frame(self) -> MultimodalFrame:
        return replace(
            self._base_frame("background_audio"),
            has_voice=True,
            voice_energy=self._jitter(0.62),
            speech_like_score=self._jitter(0.65),
            sound_direction_deg=random.uniform(45, 85),
            sound_distance_m=random.uniform(2.4, 3.6),
            face_visible=True,
            head_yaw_deg=random.uniform(-20, 20),
            gaze_to_loona_score=self._jitter(0.08),
            lip_movement_score=self._jitter(0.02),
            is_attention_target=False,
            background_audio_score=self._jitter(0.88),
        )

    def _side_talk_frame(self) -> MultimodalFrame:
        return replace(
            self._base_frame("talking_to_others"),
            has_voice=True,
            voice_energy=self._jitter(0.74),
            speech_like_score=self._jitter(0.78),
            sound_direction_deg=random.uniform(22, 46),
            sound_distance_m=random.uniform(0.8, 1.5),
            face_visible=True,
            head_yaw_deg=random.uniform(24, 48),
            gaze_to_loona_score=self._jitter(0.16),
            lip_movement_score=self._jitter(0.55),
            is_attention_target=False,
            background_audio_score=self._jitter(0.18),
        )

    def _short_sentence_frame(self) -> MultimodalFrame:
        voice_on = self._tick % 8 in {0, 1, 2, 3}
        return replace(
            self._base_frame("short_sentence"),
            has_voice=voice_on,
            voice_energy=self._jitter(0.80 if voice_on else 0.18),
            speech_like_score=self._jitter(0.82 if voice_on else 0.15),
            sound_direction_deg=random.uniform(-12, 12),
            sound_distance_m=random.uniform(0.55, 1.0),
            face_visible=True,
            head_yaw_deg=random.uniform(-10, 10),
            gaze_to_loona_score=self._jitter(0.68),
            lip_movement_score=self._jitter(0.62 if voice_on else 0.12),
            is_attention_target=True,
            background_audio_score=self._jitter(0.05),
        )
