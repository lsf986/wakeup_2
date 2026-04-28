from __future__ import annotations

import json
import socket
import time
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from loona_wakeup.models import LiveUdpConfig, MultimodalFrame


_FIELD_ALIASES = {
    "timestamp": "timestamp_ms",
    "ts": "timestamp_ms",
    "target_user_id": "user_id",
    "track_id": "target_track_id",
    "voice": "has_voice",
    "vad": "has_voice",
    "direction_deg": "sound_direction_deg",
    "source_direction_deg": "sound_direction_deg",
    "distance_m": "sound_distance_m",
    "source_distance_m": "sound_distance_m",
    "face_direction": "face_direction_deg",
    "face_direction_deg": "face_direction_deg",
    "visual_direction_deg": "face_direction_deg",
    "source_face_match": "sound_face_match_score",
    "sound_face_match": "sound_face_match_score",
    "head_yaw": "head_yaw_deg",
    "head_pitch": "head_pitch_deg",
    "gaze_score": "gaze_to_loona_score",
    "lip_score": "lip_movement_score",
    "attention_target": "is_attention_target",
    "background_score": "background_audio_score",
}


class LiveUdpAdapter(QObject):
    frame_ready = Signal(MultimodalFrame)
    status_changed = Signal(str)

    def __init__(self, config: LiveUdpConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._socket: socket.socket | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(config.poll_interval_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._config.host, self._config.port))
        self._socket.setblocking(False)
        self._timer.start()
        self.status_changed.emit(f"LIVE UDP {self._config.host}:{self._config.port}")

    def stop(self) -> None:
        self._timer.stop()
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _poll(self) -> None:
        if self._socket is None:
            return
        while True:
            try:
                payload, _address = self._socket.recvfrom(self._config.max_datagram_size)
            except BlockingIOError:
                break
            except OSError as exc:
                self.status_changed.emit(f"LIVE UDP ERROR: {exc}")
                break

            try:
                frame = frame_from_payload(json.loads(payload.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                self.status_changed.emit(f"DROP BAD FRAME: {exc}")
                continue
            self.frame_ready.emit(frame)


def frame_from_payload(payload: dict[str, Any]) -> MultimodalFrame:
    normalized = _normalize_payload(payload)
    timestamp_ms = _as_int(normalized.get("timestamp_ms"), int(time.time() * 1000))
    if timestamp_ms < 10_000_000_000:
        timestamp_ms *= 1000

    return MultimodalFrame(
        timestamp_ms=timestamp_ms,
        user_id=_as_optional_str(normalized.get("user_id")),
        has_voice=_as_bool(normalized.get("has_voice"), False),
        voice_energy=_as_float(normalized.get("voice_energy"), 0.0),
        speech_like_score=_as_float(normalized.get("speech_like_score"), 0.0),
        sound_direction_deg=_as_optional_float(normalized.get("sound_direction_deg")),
        sound_distance_m=_as_optional_float(normalized.get("sound_distance_m")),
        face_direction_deg=_as_optional_float(normalized.get("face_direction_deg")),
        sound_face_match_score=_as_float(normalized.get("sound_face_match_score"), 1.0),
        face_visible=_as_bool(normalized.get("face_visible"), False),
        head_yaw_deg=_as_optional_float(normalized.get("head_yaw_deg")),
        head_pitch_deg=_as_optional_float(normalized.get("head_pitch_deg")),
        gaze_to_loona_score=_as_float(normalized.get("gaze_to_loona_score"), 0.0),
        lip_movement_score=_as_float(normalized.get("lip_movement_score"), 0.0),
        is_attention_target=_as_bool(normalized.get("is_attention_target"), False),
        target_track_id=_as_optional_str(normalized.get("target_track_id")),
        multi_person_count=_as_int(normalized.get("multi_person_count"), 0),
        multi_person_ambiguous=_as_bool(normalized.get("multi_person_ambiguous"), False),
        scene_type=str(normalized.get("scene_type") or "live"),
        background_audio_score=_as_float(normalized.get("background_audio_score"), 0.0),
    )


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[_FIELD_ALIASES.get(key, key)] = value
    return normalized


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
