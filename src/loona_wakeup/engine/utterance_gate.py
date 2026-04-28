from __future__ import annotations

from loona_wakeup.models import MultimodalFrame, WakeupConfig


class UtteranceGate:
    def __init__(self, config: WakeupConfig) -> None:
        self._config = config
        self._frames: list[MultimodalFrame] = []
        self._started_at_ms: int | None = None
        self._last_voice_ms: int | None = None

    def push(self, frame: MultimodalFrame) -> list[MultimodalFrame] | None:
        voice_active = frame.has_voice or frame.speech_like_score >= self._config.min_voice_score

        if voice_active:
            if self._started_at_ms is None:
                self._started_at_ms = frame.timestamp_ms
            self._frames.append(frame)
            self._last_voice_ms = frame.timestamp_ms
            if frame.timestamp_ms - self._started_at_ms >= self._config.max_utterance_ms:
                return self._flush()
            return None

        if not self._frames or self._started_at_ms is None or self._last_voice_ms is None:
            return None

        silence_ms = frame.timestamp_ms - self._last_voice_ms
        if silence_ms < self._config.utterance_end_silence_ms:
            self._frames.append(frame)
            return None

        utterance_ms = self._last_voice_ms - self._started_at_ms
        if utterance_ms < self._config.min_utterance_ms:
            self.reset()
            return None
        return self._flush()

    def reset(self) -> None:
        self._frames.clear()
        self._started_at_ms = None
        self._last_voice_ms = None

    def _flush(self) -> list[MultimodalFrame]:
        frames = self._frames[:]
        self.reset()
        return frames