from loona_wakeup.engine.decision_engine import WakeupDecisionEngine
from loona_wakeup.models import MultimodalFrame, WakeupConfig, WeightConfig


def test_valid_facing_user_triggers_wakeup() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frame = MultimodalFrame(
        timestamp_ms=1,
        user_id="user_01",
        has_voice=True,
        voice_energy=0.9,
        speech_like_score=0.9,
        sound_direction_deg=0,
        sound_distance_m=0.8,
        face_visible=True,
        head_yaw_deg=0,
        gaze_to_loona_score=0.8,
        lip_movement_score=0.8,
        is_attention_target=True,
    )
    first_decision = engine.decide(frame)
    second_decision = engine.decide(frame)
    decision = engine.decide(frame)
    assert first_decision.wakeup is False
    assert second_decision.wakeup is False
    assert decision.wakeup is True


def test_single_valid_frame_waits_for_consecutive_confirmation() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide(
        MultimodalFrame(
            timestamp_ms=1,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.8,
            lip_movement_score=0.8,
            is_attention_target=True,
        )
    )
    assert decision.wakeup is False
    assert "pending_consecutive_confirmation" in decision.reject_reasons


def test_valid_utterance_triggers_after_aggregation() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.8,
            lip_movement_score=0.8,
            is_attention_target=True,
        )
        for timestamp_ms in (0, 200, 400)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is True
    assert decision.raw_scores["consecutive_wakeup_candidates"] == 0.0


def test_medium_strength_utterance_triggers_with_sensitive_defaults() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide_utterance(
        [
            MultimodalFrame(
                timestamp_ms=timestamp_ms,
                user_id="user_01",
                has_voice=True,
                voice_energy=0.55,
                speech_like_score=0.55,
                sound_direction_deg=0,
                sound_distance_m=0.8,
                face_visible=True,
                head_yaw_deg=0,
                gaze_to_loona_score=0.50,
                lip_movement_score=0.30,
                is_attention_target=True,
            )
            for timestamp_ms in (0, 160, 320)
        ]
    )
    assert decision.wakeup is True


def test_short_cough_like_burst_is_rejected_even_with_lip_motion() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.95,
            speech_like_score=0.88,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.85,
            is_attention_target=True,
        )
        for timestamp_ms in (0, 120)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "voice_burst_too_short" in decision.reject_reasons


def test_sound_face_mismatch_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=42,
            face_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.8,
            is_attention_target=True,
        )
        for timestamp_ms in (0, 160, 320)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "sound_face_mismatch" in decision.reject_reasons


def test_single_frame_sound_face_mismatch_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide(
        MultimodalFrame(
            timestamp_ms=100,
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=-42,
            face_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.8,
            is_attention_target=True,
        )
    )
    assert decision.wakeup is False
    assert "sound_face_mismatch" in decision.reject_reasons


def test_inconsistent_intent_during_voice_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            face_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=None,
            gaze_to_loona_score=0.05,
            lip_movement_score=0.8 if timestamp_ms in {160, 320} else 0.05,
            is_attention_target=False,
        )
        for timestamp_ms in (0, 160, 320, 480)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "intent_not_consistent" in decision.reject_reasons


def test_unstable_multi_person_target_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id=track_id,
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            face_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.8,
            is_attention_target=True,
            target_track_id=track_id,
            multi_person_count=2,
        )
        for timestamp_ms, track_id in ((0, "local_user_0"), (160, "local_user_0"), (320, "local_user_1"))
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "multi_person_ambiguous" in decision.reject_reasons


def test_missing_target_frames_in_multi_person_utterance_are_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="local_user_0",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            face_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.8,
            is_attention_target=True,
            target_track_id=track_id,
            multi_person_count=2,
        )
        for timestamp_ms, track_id in ((0, "local_user_0"), (160, None), (320, "local_user_0"))
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "target_not_stable" in decision.reject_reasons


def test_background_audio_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide(
        MultimodalFrame(
            timestamp_ms=1,
            has_voice=True,
            voice_energy=0.7,
            speech_like_score=0.7,
            sound_direction_deg=70,
            sound_distance_m=3.0,
            face_visible=True,
            gaze_to_loona_score=0.05,
            lip_movement_score=0.0,
            background_audio_score=0.9,
        )
    )
    assert decision.wakeup is False
    assert decision.reject_reasons


def test_voice_without_face_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide(
        MultimodalFrame(
            timestamp_ms=1,
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=False,
            gaze_to_loona_score=0.0,
            lip_movement_score=0.0,
            is_attention_target=False,
        )
    )
    assert decision.wakeup is False
    assert "face_not_visible" in decision.reject_reasons


def test_voice_without_lip_sync_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide(
        MultimodalFrame(
            timestamp_ms=1,
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.0,
            is_attention_target=True,
        )
    )
    assert decision.wakeup is False
    assert "no_lip_voice_sync" in decision.reject_reasons


def test_utterance_with_static_lips_is_rejected_even_when_voice_is_strong() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.0,
            is_attention_target=True,
        )
        for timestamp_ms in (0, 120, 240, 360)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "no_lip_voice_sync" in decision.reject_reasons


def test_utterance_with_only_one_lip_motion_spike_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    lip_scores = [0.0, 0.0, 0.8, 0.0]
    frames = [
        MultimodalFrame(
            timestamp_ms=index * 120,
            user_id="user_01",
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=lip_score,
            is_attention_target=True,
        )
        for index, lip_score in enumerate(lip_scores)
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "no_lip_voice_sync" in decision.reject_reasons


def test_multi_person_ambiguous_utterance_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide_utterance(
        [
            MultimodalFrame(
                timestamp_ms=1,
                user_id="local_user_0",
                has_voice=True,
                voice_energy=0.9,
                speech_like_score=0.9,
                sound_direction_deg=0,
                sound_distance_m=0.8,
                face_visible=True,
                head_yaw_deg=0,
                gaze_to_loona_score=0.9,
                lip_movement_score=0.9,
                is_attention_target=True,
                target_track_id="local_user_0",
                multi_person_count=2,
                multi_person_ambiguous=True,
            )
        ]
    )
    assert decision.wakeup is False
    assert "multi_person_ambiguous" in decision.reject_reasons


def test_utterance_with_track_switch_is_rejected() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    frames = [
        MultimodalFrame(
            timestamp_ms=timestamp_ms,
            user_id=track_id,
            has_voice=True,
            voice_energy=0.9,
            speech_like_score=0.9,
            sound_direction_deg=0,
            sound_distance_m=0.8,
            face_visible=True,
            head_yaw_deg=0,
            gaze_to_loona_score=0.9,
            lip_movement_score=0.9,
            is_attention_target=True,
            target_track_id=track_id,
            multi_person_count=2,
        )
        for timestamp_ms, track_id in ((0, "local_user_0"), (120, "local_user_1"), (240, "local_user_1"))
    ]
    decision = engine.decide_utterance(frames)
    assert decision.wakeup is False
    assert "multi_person_ambiguous" in decision.reject_reasons


def test_head_angle_over_30_degrees_is_rejected_even_with_voice() -> None:
    engine = WakeupDecisionEngine(WakeupConfig(), WeightConfig())
    decision = engine.decide_utterance(
        [
            MultimodalFrame(
                timestamp_ms=1,
                user_id="user_01",
                has_voice=True,
                voice_energy=0.9,
                speech_like_score=0.9,
                sound_direction_deg=0,
                sound_distance_m=0.8,
                face_visible=True,
                head_yaw_deg=31,
                gaze_to_loona_score=0.9,
                lip_movement_score=0.9,
                is_attention_target=True,
            )
        ]
    )
    assert decision.wakeup is False
    assert "head_angle_too_large" in decision.reject_reasons
