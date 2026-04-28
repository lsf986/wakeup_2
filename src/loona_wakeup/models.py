from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    MOCK = "mock"
    LOCAL = "local"
    LIVE = "live"


@dataclass(slots=True)
class MultimodalFrame:
    timestamp_ms: int
    user_id: str | None = None
    has_voice: bool = False
    voice_energy: float = 0.0
    speech_like_score: float = 0.0
    sound_direction_deg: float | None = None
    face_direction_deg: float | None = None
    sound_face_match_score: float = 1.0
    sound_distance_m: float | None = None
    face_visible: bool = False
    head_yaw_deg: float | None = None
    head_pitch_deg: float | None = None
    gaze_to_loona_score: float = 0.0
    lip_movement_score: float = 0.0
    is_attention_target: bool = False
    target_track_id: str | None = None
    multi_person_count: int = 0
    multi_person_ambiguous: bool = False
    utterance_voice_ms: int = 0
    utterance_voice_frame_count: int = 0
    intent_consistency_score: float = 1.0
    target_stability_score: float = 1.0
    human_conversation_score: float = 0.0
    transcript: str = ""
    text_completeness_score: float = 1.0
    direct_address_score: float = 0.0
    self_talk_score: float = 0.0
    scene_type: str = "unknown"
    background_audio_score: float = 0.0


@dataclass(slots=True)
class WakeupDecision:
    timestamp_ms: int
    wakeup: bool
    confidence: float
    target_user_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        state = "WAKEUP" if self.wakeup else "IDLE"
        reason_text = ", ".join(self.reasons or self.reject_reasons[:2])
        return f"{state}  conf={self.confidence:.2f}  {reason_text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "wakeup": self.wakeup,
            "confidence": self.confidence,
            "target_user_id": self.target_user_id,
            "reasons": self.reasons,
            "reject_reasons": self.reject_reasons,
            "raw_scores": self.raw_scores,
        }


@dataclass(slots=True)
class RuntimeConfig:
    mode: RunMode = RunMode.MOCK
    ui_refresh_ms: int = 100
    decision_interval_ms: int = 100


@dataclass(slots=True)
class WakeupConfig:
    min_confidence: float = 0.66
    min_voice_score: float = 0.38
    max_distance_m: float = 2.0
    min_gaze_score: float = 0.38
    min_lip_score: float = 0.24
    min_visual_intent_score: float = 0.45
    max_head_yaw_deg: float = 30.0
    require_face_visible: bool = True
    require_lip_sync: bool = True
    min_consecutive_wakeup_frames: int = 3
    utterance_end_silence_ms: int = 350
    min_utterance_ms: int = 220
    min_wakeup_voice_ms: int = 320
    min_wakeup_voice_frames: int = 3
    min_intent_consistency_score: float = 0.45
    min_target_stability_score: float = 0.75
    min_sound_face_match_score: float = 0.45
    max_human_conversation_score: float = 0.62
    min_text_completeness_score: float = 0.45
    min_direct_address_score: float = 0.30
    max_self_talk_score: float = 0.62
    max_utterance_ms: int = 8000
    decision_window_ms: int = 1200
    cooldown_ms: int = 1500


@dataclass(slots=True)
class WeightConfig:
    voice_score: float = 0.25
    sound_position: float = 0.20
    distance_score: float = 0.15
    head_pose_score: float = 0.15
    gaze_score: float = 0.15
    lip_score: float = 0.20
    attention_bonus: float = 0.10
    background_penalty: float = -0.20


@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = "logs/decisions.jsonl"


@dataclass(slots=True)
class LiveUdpConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    poll_interval_ms: int = 30
    max_datagram_size: int = 65535


@dataclass(slots=True)
class LocalInputConfig:
    camera_index: int = 0
    frame_interval_ms: int = 80
    audio_sample_rate: int = 16000
    audio_block_size: int = 1024
    min_audio_energy: float = 0.008
    min_mouth_motion: float = 0.008


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wakeup: WakeupConfig = field(default_factory=WakeupConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    live_udp: LiveUdpConfig = field(default_factory=LiveUdpConfig)
    local_input: LocalInputConfig = field(default_factory=LocalInputConfig)
