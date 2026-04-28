from loona_wakeup.engine.utterance_gate import UtteranceGate
from loona_wakeup.models import MultimodalFrame, WakeupConfig


def _frame(timestamp_ms: int, has_voice: bool) -> MultimodalFrame:
    return MultimodalFrame(
        timestamp_ms=timestamp_ms,
        has_voice=has_voice,
        voice_energy=0.8 if has_voice else 0.0,
        speech_like_score=0.8 if has_voice else 0.0,
    )


def test_gate_does_not_emit_while_user_is_speaking() -> None:
    gate = UtteranceGate(WakeupConfig(utterance_end_silence_ms=700, min_utterance_ms=300))
    assert gate.push(_frame(0, True)) is None
    assert gate.push(_frame(200, True)) is None
    assert gate.push(_frame(400, True)) is None


def test_gate_emits_after_sentence_end_silence() -> None:
    gate = UtteranceGate(WakeupConfig(utterance_end_silence_ms=700, min_utterance_ms=300))
    gate.push(_frame(0, True))
    gate.push(_frame(200, True))
    gate.push(_frame(400, True))
    assert gate.push(_frame(900, False)) is None
    utterance = gate.push(_frame(1200, False))
    assert utterance is not None
    assert [frame.timestamp_ms for frame in utterance] == [0, 200, 400, 900]


def test_default_gate_emits_quickly_after_sentence_end() -> None:
    gate = UtteranceGate(WakeupConfig())
    gate.push(_frame(0, True))
    gate.push(_frame(240, True))
    assert gate.push(_frame(520, False)) is None
    utterance = gate.push(_frame(600, False))
    assert utterance is not None
    assert [frame.timestamp_ms for frame in utterance] == [0, 240, 520]


def test_gate_drops_too_short_voice_burst() -> None:
    gate = UtteranceGate(WakeupConfig(utterance_end_silence_ms=700, min_utterance_ms=300))
    gate.push(_frame(0, True))
    gate.push(_frame(100, True))
    assert gate.push(_frame(900, False)) is None
