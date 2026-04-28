from __future__ import annotations

from math import isfinite

from loona_wakeup.engine.utterance_text_analyzer import analyze_utterance_text
from loona_wakeup.models import MultimodalFrame, WakeupConfig, WakeupDecision, WeightConfig


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not isfinite(value):
        return low
    return max(low, min(high, value))


def _direction_score(direction_deg: float | None) -> float:
    if direction_deg is None:
        return 0.0
    absolute = abs(direction_deg)
    if absolute <= 10:
        return 1.0
    if absolute >= 60:
        return 0.0
    return _clamp(1.0 - ((absolute - 10.0) / 50.0))


def _distance_score(distance_m: float | None, max_distance_m: float) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= 0.4:
        return 1.0
    if distance_m >= max_distance_m:
        return 0.0
    return _clamp(1.0 - ((distance_m - 0.4) / (max_distance_m - 0.4)))


def _head_pose_score(yaw_deg: float | None, face_visible: bool) -> float:
    if not face_visible or yaw_deg is None:
        return 0.0
    absolute = abs(yaw_deg)
    if absolute <= 12:
        return 1.0
    if absolute >= 55:
        return 0.0
    return _clamp(1.0 - ((absolute - 12.0) / 43.0))


def _sound_face_match_score(sound_direction_deg: float | None, face_direction_deg: float | None) -> float:
    if sound_direction_deg is None or face_direction_deg is None:
        return 1.0
    gap = abs(sound_direction_deg - face_direction_deg)
    if gap <= 8.0:
        return 1.0
    if gap >= 45.0:
        return 0.0
    return _clamp(1.0 - ((gap - 8.0) / 37.0))


class WakeupDecisionEngine:
    def __init__(self, wakeup_config: WakeupConfig, weights: WeightConfig) -> None:
        self.wakeup_config = wakeup_config
        self.weights = weights
        self._consecutive_wakeup_candidates = 0

    def decide(self, frame: MultimodalFrame) -> WakeupDecision:
        return self._decide_frame(frame, require_consecutive=True)

    def decide_utterance(self, frames: list[MultimodalFrame]) -> WakeupDecision:
        if not frames:
            return WakeupDecision(timestamp_ms=0, wakeup=False, confidence=0.0, reject_reasons=["empty_utterance"])
        return self._decide_frame(self._aggregate_utterance(frames), require_consecutive=False)

    def _decide_frame(self, frame: MultimodalFrame, require_consecutive: bool) -> WakeupDecision:
        raw_scores = self._score_frame(frame)
        reject_reasons = self._hard_rejects(frame, raw_scores)
        reasons = self._reasons(frame, raw_scores)

        weighted = (
            raw_scores["voice_score"] * self.weights.voice_score
            + raw_scores["sound_position"] * self.weights.sound_position
            + raw_scores["distance_score"] * self.weights.distance_score
            + raw_scores["head_pose_score"] * self.weights.head_pose_score
            + raw_scores["gaze_score"] * self.weights.gaze_score
            + raw_scores["lip_score"] * self.weights.lip_score
            + raw_scores["attention_score"] * self.weights.attention_bonus
            + raw_scores["background_audio_score"] * self.weights.background_penalty
        )
        max_positive = (
            self.weights.voice_score
            + self.weights.sound_position
            + self.weights.distance_score
            + self.weights.head_pose_score
            + self.weights.gaze_score
            + self.weights.lip_score
            + self.weights.attention_bonus
        )
        confidence = _clamp(weighted / max_positive if max_positive else 0.0)
        is_candidate = not reject_reasons and confidence >= self.wakeup_config.min_confidence

        if is_candidate and require_consecutive:
            self._consecutive_wakeup_candidates += 1
        else:
            self._consecutive_wakeup_candidates = 0

        wakeup = is_candidate
        if require_consecutive:
            wakeup = is_candidate and self._consecutive_wakeup_candidates >= self.wakeup_config.min_consecutive_wakeup_frames

        if not wakeup and not reject_reasons:
            if is_candidate and require_consecutive:
                reject_reasons.append("pending_consecutive_confirmation")
            else:
                reject_reasons.append("confidence_below_threshold")

        raw_scores["consecutive_wakeup_candidates"] = float(self._consecutive_wakeup_candidates)

        return WakeupDecision(
            timestamp_ms=frame.timestamp_ms,
            wakeup=wakeup,
            confidence=confidence,
            target_user_id=frame.user_id if wakeup else None,
            reasons=reasons if wakeup else [],
            reject_reasons=[] if wakeup else reject_reasons,
            raw_scores=raw_scores,
        )

    def _aggregate_utterance(self, frames: list[MultimodalFrame]) -> MultimodalFrame:
        voice_frames = [frame for frame in frames if frame.has_voice or frame.speech_like_score >= self.wakeup_config.min_voice_score]
        source_frames = voice_frames or frames
        voice_duration_ms = self._voice_duration_ms(voice_frames)
        best_voice = max(source_frames, key=lambda frame: (frame.speech_like_score, frame.voice_energy))
        best_visual = max(source_frames, key=lambda frame: frame.gaze_to_loona_score + frame.lip_movement_score)
        strongest_head_yaw = max(
            (frame.head_yaw_deg for frame in source_frames if frame.head_yaw_deg is not None),
            key=abs,
            default=None,
        )
        best_distance = min(
            (frame.sound_distance_m for frame in source_frames if frame.sound_distance_m is not None),
            default=None,
        )
        target_track_ids = {frame.target_track_id for frame in source_frames if frame.target_track_id}
        target_track_id = next(iter(target_track_ids)) if len(target_track_ids) == 1 else None
        multi_person_count = max((frame.multi_person_count for frame in source_frames), default=0)
        multi_person_ambiguous = any(frame.multi_person_ambiguous for frame in source_frames) or len(target_track_ids) > 1
        stable_lip_score = self._stable_lip_score(source_frames)
        intent_consistency_score = self._intent_consistency_score(source_frames)
        target_stability_score = self._target_stability_score(source_frames, target_track_id)
        sound_face_match_score = self._aggregate_sound_face_match_score(source_frames)
        human_conversation_score = self._aggregate_human_conversation_score(source_frames)
        transcript = self._aggregate_transcript(source_frames)
        text_scores = self._aggregate_text_scores(source_frames, transcript)
        return MultimodalFrame(
            timestamp_ms=frames[-1].timestamp_ms,
            user_id=best_visual.user_id or best_voice.user_id,
            has_voice=any(frame.has_voice for frame in source_frames),
            voice_energy=max(frame.voice_energy for frame in source_frames),
            speech_like_score=max(frame.speech_like_score for frame in source_frames),
            sound_direction_deg=best_voice.sound_direction_deg,
            face_direction_deg=best_visual.face_direction_deg,
            sound_face_match_score=sound_face_match_score,
            sound_distance_m=best_distance,
            face_visible=any(frame.face_visible for frame in source_frames),
            head_yaw_deg=strongest_head_yaw if strongest_head_yaw is not None else best_visual.head_yaw_deg,
            head_pitch_deg=best_visual.head_pitch_deg,
            gaze_to_loona_score=max(frame.gaze_to_loona_score for frame in source_frames),
            lip_movement_score=stable_lip_score,
            is_attention_target=any(frame.is_attention_target for frame in source_frames),
            target_track_id=target_track_id,
            multi_person_count=multi_person_count,
            multi_person_ambiguous=multi_person_ambiguous,
            utterance_voice_ms=voice_duration_ms,
            utterance_voice_frame_count=len(voice_frames),
            intent_consistency_score=intent_consistency_score,
            target_stability_score=target_stability_score,
            human_conversation_score=human_conversation_score,
            transcript=transcript,
            text_completeness_score=text_scores["text_completeness_score"],
            direct_address_score=text_scores["direct_address_score"],
            self_talk_score=text_scores["self_talk_score"],
            scene_type="utterance_aggregate",
            background_audio_score=min(frame.background_audio_score for frame in source_frames),
        )

    def _voice_duration_ms(self, voice_frames: list[MultimodalFrame]) -> int:
        if len(voice_frames) < 2:
            return 0
        return max(0, voice_frames[-1].timestamp_ms - voice_frames[0].timestamp_ms)

    def _intent_consistency_score(self, frames: list[MultimodalFrame]) -> float:
        voice_frames = [frame for frame in frames if frame.has_voice or frame.speech_like_score >= self.wakeup_config.min_voice_score]
        if not voice_frames:
            return 0.0
        scores = []
        for frame in voice_frames:
            head_score = _head_pose_score(frame.head_yaw_deg, frame.face_visible)
            attention_score = 1.0 if frame.is_attention_target else 0.0
            visual_attention = max(_clamp(frame.gaze_to_loona_score), attention_score, head_score)
            lip_score = _clamp(frame.lip_movement_score)
            scores.append((visual_attention * 0.65) + (lip_score * 0.35))
        return sum(scores) / len(scores)

    def _target_stability_score(self, frames: list[MultimodalFrame], target_track_id: str | None) -> float:
        voice_frames = [frame for frame in frames if frame.has_voice or frame.speech_like_score >= self.wakeup_config.min_voice_score]
        if not voice_frames or target_track_id is None:
            return 1.0
        stable_frames = sum(1 for frame in voice_frames if frame.target_track_id == target_track_id)
        return stable_frames / len(voice_frames)

    def _aggregate_sound_face_match_score(self, frames: list[MultimodalFrame]) -> float:
        voice_frames = [frame for frame in frames if frame.has_voice or frame.speech_like_score >= self.wakeup_config.min_voice_score]
        source_frames = voice_frames or frames
        scores = [
            min(_clamp(frame.sound_face_match_score), _sound_face_match_score(frame.sound_direction_deg, frame.face_direction_deg))
            for frame in source_frames
        ]
        return sum(scores) / len(scores) if scores else 1.0

    def _aggregate_human_conversation_score(self, frames: list[MultimodalFrame]) -> float:
        voice_frames = [frame for frame in frames if frame.has_voice or frame.speech_like_score >= self.wakeup_config.min_voice_score]
        source_frames = voice_frames or frames
        if not source_frames:
            return 0.0
        return sum(_clamp(frame.human_conversation_score) for frame in source_frames) / len(source_frames)

    def _aggregate_transcript(self, frames: list[MultimodalFrame]) -> str:
        transcripts = [frame.transcript.strip() for frame in frames if frame.transcript.strip()]
        return max(transcripts, key=len, default="")

    def _aggregate_text_scores(self, frames: list[MultimodalFrame], transcript: str) -> dict[str, float]:
        direct_address_score = max((_clamp(frame.direct_address_score) for frame in frames), default=0.0)
        self_talk_score = max((_clamp(frame.self_talk_score) for frame in frames), default=0.0)
        completeness_scores = [frame.text_completeness_score for frame in frames if frame.text_completeness_score != 1.0]
        text_completeness_score = min((_clamp(score) for score in completeness_scores), default=1.0)

        if transcript:
            analysis = analyze_utterance_text(transcript)
            direct_address_score = max(direct_address_score, analysis.direct_address_score)
            self_talk_score = max(self_talk_score, analysis.self_talk_score)
            text_completeness_score = min(text_completeness_score, analysis.completeness_score)

        return {
            "text_completeness_score": text_completeness_score,
            "direct_address_score": direct_address_score,
            "self_talk_score": self_talk_score,
        }

    def _stable_lip_score(self, frames: list[MultimodalFrame]) -> float:
        lip_scores = [_clamp(frame.lip_movement_score) for frame in frames]
        if not lip_scores:
            return 0.0
        if len(lip_scores) <= 2:
            return max(lip_scores)

        active_count = sum(score >= self.wakeup_config.min_lip_score for score in lip_scores)
        required_active_count = max(2, (len(lip_scores) + 3) // 4)
        if active_count < required_active_count:
            return min(max(lip_scores), self.wakeup_config.min_lip_score * 0.75)

        top_count = max(1, len(lip_scores) // 3)
        top_scores = sorted(lip_scores, reverse=True)[:top_count]
        return sum(top_scores) / len(top_scores)

    def _score_frame(self, frame: MultimodalFrame) -> dict[str, float]:
        return {
            "voice_score": _clamp((frame.voice_energy * 0.35) + (frame.speech_like_score * 0.65)),
            "sound_position": _direction_score(frame.sound_direction_deg),
            "sound_face_match_score": min(
                _clamp(frame.sound_face_match_score),
                _sound_face_match_score(frame.sound_direction_deg, frame.face_direction_deg),
            ),
            "distance_score": _distance_score(frame.sound_distance_m, self.wakeup_config.max_distance_m),
            "head_pose_score": _head_pose_score(frame.head_yaw_deg, frame.face_visible),
            "gaze_score": _clamp(frame.gaze_to_loona_score),
            "lip_score": _clamp(frame.lip_movement_score),
            "intent_consistency_score": _clamp(frame.intent_consistency_score),
            "target_stability_score": _clamp(frame.target_stability_score),
            "human_conversation_score": _clamp(frame.human_conversation_score),
            "text_completeness_score": _clamp(frame.text_completeness_score),
            "direct_address_score": _clamp(frame.direct_address_score),
            "self_talk_score": _clamp(frame.self_talk_score),
            "attention_score": 1.0 if frame.is_attention_target else 0.0,
            "background_audio_score": _clamp(frame.background_audio_score),
        }

    def _hard_rejects(self, frame: MultimodalFrame, raw_scores: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if not frame.has_voice or raw_scores["voice_score"] < self.wakeup_config.min_voice_score:
            reasons.append("no_reliable_voice")
        if frame.scene_type == "utterance_aggregate":
            if frame.utterance_voice_frame_count < self.wakeup_config.min_wakeup_voice_frames:
                reasons.append("voice_burst_too_short")
            elif frame.utterance_voice_ms < self.wakeup_config.min_wakeup_voice_ms:
                reasons.append("voice_burst_too_short")
            if raw_scores["intent_consistency_score"] < self.wakeup_config.min_intent_consistency_score:
                reasons.append("intent_not_consistent")
            if frame.multi_person_count > 1 and raw_scores["target_stability_score"] < self.wakeup_config.min_target_stability_score:
                reasons.append("target_not_stable")
            if frame.multi_person_count > 1 and raw_scores["human_conversation_score"] > self.wakeup_config.max_human_conversation_score:
                reasons.append("human_conversation_detected")
            if self._has_text_signal(frame):
                if raw_scores["text_completeness_score"] < self.wakeup_config.min_text_completeness_score:
                    reasons.append("incomplete_utterance_text")
                if raw_scores["self_talk_score"] > self.wakeup_config.max_self_talk_score:
                    reasons.append("self_talk_detected")
                if raw_scores["direct_address_score"] < self.wakeup_config.min_direct_address_score and raw_scores["self_talk_score"] >= 0.30:
                    reasons.append("no_direct_address")
        if raw_scores["sound_face_match_score"] < self.wakeup_config.min_sound_face_match_score:
            reasons.append("sound_face_mismatch")
        if frame.multi_person_ambiguous:
            reasons.append("multi_person_ambiguous")
        if self.wakeup_config.require_face_visible and not frame.face_visible:
            reasons.append("face_not_visible")
        if frame.head_yaw_deg is not None and abs(frame.head_yaw_deg) > self.wakeup_config.max_head_yaw_deg:
            reasons.append("head_angle_too_large")
        if frame.sound_distance_m is not None and frame.sound_distance_m > self.wakeup_config.max_distance_m:
            reasons.append("distance_too_far")
        if raw_scores["background_audio_score"] > 0.70 and raw_scores["lip_score"] < self.wakeup_config.min_lip_score:
            reasons.append("background_audio_without_lip_sync")
        if not frame.face_visible and raw_scores["sound_position"] < 0.50:
            reasons.append("no_visible_target")
        visual_intent_score = max(
            raw_scores["head_pose_score"],
            raw_scores["gaze_score"],
            raw_scores["lip_score"],
            raw_scores["attention_score"],
        )
        if visual_intent_score < self.wakeup_config.min_visual_intent_score:
            reasons.append("no_intent_signal")
        if self.wakeup_config.require_lip_sync and raw_scores["lip_score"] < self.wakeup_config.min_lip_score:
            reasons.append("no_lip_voice_sync")
        return reasons

    def _has_text_signal(self, frame: MultimodalFrame) -> bool:
        return bool(frame.transcript.strip()) or frame.direct_address_score > 0.0 or frame.self_talk_score > 0.0 or frame.text_completeness_score != 1.0

    def _reasons(self, frame: MultimodalFrame, raw_scores: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if raw_scores["voice_score"] >= self.wakeup_config.min_voice_score:
            reasons.append("voice_near")
        if raw_scores["sound_position"] >= 0.65:
            reasons.append("sound_from_user_direction")
        if raw_scores["distance_score"] >= 0.60:
            reasons.append("distance_valid")
        if raw_scores["head_pose_score"] >= 0.60:
            reasons.append("head_facing_loona")
        if raw_scores["gaze_score"] >= self.wakeup_config.min_gaze_score:
            reasons.append("gaze_to_loona")
        if raw_scores["lip_score"] >= self.wakeup_config.min_lip_score:
            reasons.append("lip_synced")
        if frame.is_attention_target:
            reasons.append("attention_target")
        return reasons
