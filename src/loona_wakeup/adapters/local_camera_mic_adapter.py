from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sounddevice as sd
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage

from loona_wakeup.models import LocalInputConfig, MultimodalFrame

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
except ImportError:  # pragma: no cover - optional high-accuracy backend
    mp = None
    vision = None
    BaseOptions = None


FACE_OVAL_POINTS = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
LEFT_EYE_POINTS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_POINTS = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
OUTER_LIP_POINTS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP_POINTS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIP_OPENING_PAIRS = [(13, 14), (82, 87), (81, 178), (80, 88), (312, 317), (311, 402), (310, 318)]
HUD_GREEN = (78, 255, 166)
HUD_RED = (255, 82, 104)
HUD_TEXT = (230, 237, 243)
HUD_DIM = (96, 110, 124)
MAX_HEAD_YAW_DEG = 30.0
MAX_HEAD_PITCH_DEG = 30.0
EYE_OCCLUSION_EVIDENCE_THRESHOLD = 0.06
MOUTH_OCCLUSION_EVIDENCE_THRESHOLD = 0.05
MULTI_PERSON_SELECTION_MARGIN = 0.08
MULTI_PERSON_LIP_DOMINANCE_MARGIN = 0.012
GAZE_ENTER_THRESHOLD = 0.55
GAZE_EXIT_THRESHOLD = 0.48
TRACK_MATCH_DISTANCE_RATIO = 0.38
TRACK_MAX_STALE_FRAMES = 8
LIP_OPEN_RATIO_DEADZONE = 0.012
LIP_OPEN_RATIO_SCORE_SCALE = 6.5


def _clamp_local(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _direction_score_local(direction_deg: float | None) -> float:
    if direction_deg is None:
        return 0.0
    absolute = abs(direction_deg)
    if absolute <= 10.0:
        return 1.0
    if absolute >= 45.0:
        return 0.0
    return _clamp_local(1.0 - ((absolute - 10.0) / 35.0))


def _head_facing_score_local(head_yaw_deg: float | None) -> float:
    if head_yaw_deg is None:
        return 0.0
    absolute = abs(head_yaw_deg)
    if absolute <= 10.0:
        return 1.0
    if absolute >= MAX_HEAD_YAW_DEG:
        return 0.0
    return _clamp_local(1.0 - ((absolute - 10.0) / (MAX_HEAD_YAW_DEG - 10.0)))


def _distance_score_local(distance_m: float | None) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= 0.7:
        return 1.0
    if distance_m >= 2.2:
        return 0.0
    return _clamp_local(1.0 - ((distance_m - 0.7) / 1.5))


class LocalCameraMicAdapter(QObject):
    frame_ready = Signal(MultimodalFrame)
    preview_ready = Signal(QImage)
    status_changed = Signal(str)

    def __init__(self, config: LocalInputConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._audio_stream: sd.InputStream | None = None
        self._audio_energy = 0.0
        self._audio_speech_quality = 1.0
        self._noise_floor = 0.01
        self._prev_mouth_gray: np.ndarray | None = None
        self._prev_lip_open_ratio: float | None = None
        self._prev_lip_open_ratios: dict[str, float] = {}
        self._mouth_motion_history: deque[float] = deque(maxlen=5)
        self._mouth_motion_histories: dict[str, deque[float]] = {}
        self._gaze_histories: dict[str, deque[float]] = {}
        self._gaze_states: dict[str, bool] = {}
        self._tracks: dict[str, dict[str, Any]] = {}
        self._next_track_index = 0
        self._frame_index = 0
        self._last_mesh_timestamp_ms = 0
        self._timer = QTimer(self)
        self._timer.setInterval(config.frame_interval_ms)
        self._timer.timeout.connect(self._poll)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        eyeglasses_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self._eyeglasses_eye_cascade = cv2.CascadeClassifier(eyeglasses_cascade_path)
        self._face_mesh: Any | None = None
        self._hand_landmarker: Any | None = None
        model_path = Path(__file__).resolve().parents[3] / "assets" / "models" / "face_landmarker.task"
        if mp is not None and vision is not None and BaseOptions is not None and model_path.exists():
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=3,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True,
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)
        hand_model_path = Path(__file__).resolve().parents[3] / "assets" / "models" / "hand_landmarker.task"
        if mp is not None and vision is not None and BaseOptions is not None and hand_model_path.exists():
            hand_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self._config.camera_index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self.status_changed.emit("LOCAL CAMERA unavailable")
        else:
            self.status_changed.emit("LOCAL camera+mic")

        try:
            self._audio_stream = sd.InputStream(
                channels=1,
                samplerate=self._config.audio_sample_rate,
                blocksize=self._config.audio_block_size,
                callback=self._on_audio,
            )
            self._audio_stream.start()
        except Exception as exc:  # pragma: no cover - depends on host audio device
            self.status_changed.emit(f"LOCAL audio unavailable: {exc}")

        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        samples = np.asarray(indata, dtype=np.float32).reshape(-1)
        energy = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if energy < self._noise_floor * 2.0:
            self._noise_floor = (self._noise_floor * 0.98) + (energy * 0.02)
        self._audio_energy = energy
        self._audio_speech_quality = self._speech_quality_score(samples, self._config.audio_sample_rate)

    def _poll(self) -> None:
        if self._capture is None or not self._capture.isOpened():
            self._emit_empty_frame()
            return

        ok, bgr = self._capture.read()
        if not ok:
            self._emit_empty_frame()
            return
        self._frame_index += 1

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mesh_result = self._detect_face_mesh(rgb)
        faces = [] if mesh_result is not None else self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        face = mesh_result["face"] if mesh_result is not None else self._largest_face(faces)

        face_visible = face is not None
        head_yaw_deg: float | None = None
        head_pitch_deg: float | None = None
        gaze_score = 0.0
        lip_motion = 0.0
        distance_m: float | None = None
        direction_deg: float | None = None
        human_conversation_score = 0.0

        if mesh_result is not None:
            direction_deg = mesh_result["direction_deg"]
            head_yaw_deg = mesh_result["head_yaw_deg"]
            head_pitch_deg = mesh_result.get("head_pitch_deg")
            gaze_score = mesh_result["gaze_score"]
            distance_m = mesh_result["distance_m"]
            lip_motion = mesh_result["lip_motion"]
            human_conversation_score = mesh_result.get("human_conversation_score", 0.0)
            self._draw_face_mesh_overlays(rgb, mesh_result)
            if mesh_result.get("multi_person_ambiguous"):
                gaze_score = 0.0
                lip_motion = 0.0
        elif face is not None:
            x, y, w, h = face
            center_x = x + (w / 2.0)
            frame_center_x = rgb.shape[1] / 2.0
            normalized_offset = (center_x - frame_center_x) / frame_center_x
            direction_deg = float(normalized_offset * 35.0)
            head_yaw_deg = direction_deg
            face_center_score = max(0.0, min(1.0, 1.0 - abs(normalized_offset)))
            eyes = self._detect_eyes(gray, face)
            gaze_score = self._estimate_gaze_score(face_center_score, eyes)
            distance_m = max(0.35, min(3.0, 120.0 / max(float(w), 1.0)))
            lip_motion = self._estimate_mouth_motion(gray, face)
            self._draw_detection_overlays(rgb, face, lip_motion, eyes, gaze_score)
        else:
            self._prev_lip_open_ratio = None
            self._draw_missing_face(rgb)

        image = self._to_qimage(rgb)
        self.preview_ready.emit(image)

        speech_like = self._speech_score()
        has_voice = speech_like >= 0.45
        lip_score = max(lip_motion, 0.65 if has_voice and lip_motion >= self._config.min_mouth_motion else lip_motion)

        frame = MultimodalFrame(
            timestamp_ms=int(time.time() * 1000),
            user_id=mesh_result.get("candidate_id") if mesh_result is not None and face_visible else ("local_user" if face_visible else None),
            has_voice=has_voice,
            voice_energy=min(1.0, self._audio_energy / max(self._config.min_audio_energy * 4.0, 1e-6)),
            speech_like_score=speech_like,
            sound_direction_deg=direction_deg,
            face_direction_deg=direction_deg,
            sound_face_match_score=1.0,
            sound_distance_m=distance_m,
            face_visible=face_visible,
            head_yaw_deg=head_yaw_deg,
            head_pitch_deg=head_pitch_deg,
            gaze_to_loona_score=gaze_score,
            lip_movement_score=lip_score,
            is_attention_target=face_visible and gaze_score >= 0.55,
            target_track_id=mesh_result.get("candidate_id") if mesh_result is not None else None,
            multi_person_count=int(mesh_result.get("multi_person_count", 0)) if mesh_result is not None else 0,
            multi_person_ambiguous=bool(mesh_result.get("multi_person_ambiguous", False)) if mesh_result is not None else False,
            human_conversation_score=human_conversation_score,
            scene_type="local_camera_mic",
            background_audio_score=0.0 if face_visible else min(1.0, speech_like),
        )
        self.frame_ready.emit(frame)

    def _emit_empty_frame(self) -> None:
        speech_score = self._speech_score()
        voice_energy = min(1.0, self._audio_energy / max(self._config.min_audio_energy * 4.0, 1e-6))
        self.frame_ready.emit(
            MultimodalFrame(
                timestamp_ms=int(time.time() * 1000),
                has_voice=speech_score >= 0.45,
                voice_energy=voice_energy,
                speech_like_score=speech_score,
                scene_type="local_no_camera",
                background_audio_score=speech_score,
            )
        )

    def _speech_score(self) -> float:
        adjusted = max(0.0, self._audio_energy - (self._noise_floor * 1.5))
        energy_score = max(0.0, min(1.0, adjusted / max(self._config.min_audio_energy, 1e-6)))
        return energy_score * self._audio_speech_quality

    def _speech_quality_score(self, samples: np.ndarray, sample_rate: int) -> float:
        if samples.size < 32:
            return 0.0
        centered = samples.astype(np.float32) - float(np.mean(samples))
        energy = float(np.sqrt(np.mean(np.square(centered))))
        if energy <= 1e-6:
            return 0.0

        signs = np.signbit(centered)
        zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1])) if centered.size > 1 else 0.0

        windowed = centered * np.hanning(centered.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed)) + 1e-9
        spectral_flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        frequencies = np.fft.rfftfreq(centered.size, d=1.0 / float(sample_rate))
        centroid = float(np.sum(frequencies * spectrum) / np.sum(spectrum)) if np.sum(spectrum) > 0 else 0.0
        centroid_ratio = centroid / max(sample_rate / 2.0, 1.0)
        periodicity_score = self._periodicity_score(centered, sample_rate)
        peak_ratio = float(np.max(np.abs(centered)) / max(energy, 1e-6))
        impulse_penalty = _clamp_local((peak_ratio - 5.0) / 7.0)

        tonal_score = 1.0 - _clamp_local((spectral_flatness - 0.35) / 0.35)
        voicing_score = 1.0 - _clamp_local((zero_crossing_rate - 0.16) / 0.16)
        low_band_score = 1.0 - _clamp_local((centroid_ratio - 0.45) / 0.30)
        return _clamp_local(
            0.10
            + (0.30 * tonal_score)
            + (0.18 * voicing_score)
            + (0.14 * low_band_score)
            + (0.32 * periodicity_score)
            - (0.18 * impulse_penalty)
        )

    def _periodicity_score(self, samples: np.ndarray, sample_rate: int) -> float:
        if samples.size < 64:
            return 0.0
        normalized = samples / max(float(np.sqrt(np.mean(np.square(samples)))), 1e-6)
        min_lag = max(1, int(sample_rate / 350.0))
        max_lag = min(normalized.size - 1, int(sample_rate / 70.0))
        if max_lag <= min_lag:
            return 0.0
        correlations = []
        for lag in range(min_lag, max_lag + 1):
            current = normalized[:-lag]
            shifted = normalized[lag:]
            denominator = max(float(np.linalg.norm(current) * np.linalg.norm(shifted)), 1e-6)
            correlations.append(float(np.dot(current, shifted) / denominator))
        best = max(correlations, default=0.0)
        return _clamp_local((best - 0.18) / 0.42)

    def _detect_face_mesh(self, rgb: np.ndarray) -> dict[str, Any] | None:
        if self._face_mesh is None or mp is None:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_mesh_timestamp_ms + 1)
        self._last_mesh_timestamp_ms = timestamp_ms
        result = self._face_mesh.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._detect_hands(mp_image, timestamp_ms)
        if not result.face_landmarks:
            self._prev_lip_open_ratio = None
            self._prev_lip_open_ratios.clear()
            self._mouth_motion_histories.clear()
            self._gaze_histories.clear()
            self._gaze_states.clear()
            self._tracks.clear()
            return None

        height, width = rgb.shape[:2]
        used_track_ids: set[str] = set()
        candidates = [
            self._build_face_mesh_candidate(result, face_index, rgb, hand_result, width, height, used_track_ids)
            for face_index in range(len(result.face_landmarks))
        ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return None
        selected, ambiguous = self._select_face_mesh_candidate(candidates)
        if selected is None:
            selected = max(candidates, key=lambda candidate: candidate["candidate_score"])
            selected = selected.copy()
            selected["gaze_score"] = 0.0
            selected["lip_motion"] = 0.0

        selected["candidates"] = candidates
        selected["multi_person_count"] = len(candidates)
        selected["multi_person_ambiguous"] = ambiguous
        selected["human_conversation_score"] = self._human_conversation_score(candidates, selected, ambiguous)
        self._drop_stale_tracks(used_track_ids)
        return selected

    def _build_face_mesh_candidate(
        self,
        result: Any,
        face_index: int,
        rgb: np.ndarray,
        hand_result: Any | None,
        width: int,
        height: int,
        used_track_ids: set[str],
    ) -> dict[str, Any] | None:
        landmarks = result.face_landmarks[face_index]
        points = self._landmark_points(landmarks, width, height, range(len(landmarks)))
        face = self._points_bbox(points)
        face_center_x = face[0] + (face[2] / 2.0)
        frame_center_x = width / 2.0
        normalized_offset = (face_center_x - frame_center_x) / frame_center_x
        face_center_score = max(0.0, min(1.0, 1.0 - abs(normalized_offset)))
        matrix_yaw_deg = self._face_matrix_yaw_deg(result, face_index=face_index)
        matrix_pitch_deg = self._face_matrix_pitch_deg(result, face_index=face_index)
        head_yaw_deg = matrix_yaw_deg if matrix_yaw_deg is not None else float(normalized_offset * 35.0)
        head_pitch_deg = matrix_pitch_deg
        side_profile = self._is_side_profile(landmarks, width, height, head_yaw_deg)
        candidate_id = self._assign_track_id(face, used_track_ids)

        eye_occlusion = self._eye_occlusion_state(landmarks, rgb, hand_result, width, height)
        mouth_occluded = self._mouth_is_occluded(landmarks, rgb, hand_result, width, height)
        if mouth_occluded:
            self._prev_lip_open_ratios.pop(candidate_id, None)
            self._mouth_motion_histories.pop(candidate_id, None)
            lip_motion = 0.0
        else:
            lip_motion = self._estimate_mesh_lip_motion(landmarks, width, height, candidate_id=candidate_id)
        raw_gaze_score = self._estimate_mesh_gaze_score(landmarks, width, height, face_center_score, eye_occlusion)
        gaze_score = self._stabilized_gaze_score(candidate_id, raw_gaze_score, eye_occlusion)
        gaze_active = self._stable_gaze_state(candidate_id, gaze_score)
        distance_m = max(0.35, min(3.0, 120.0 / max(float(face[2]), 1.0)))
        direction_deg = float(normalized_offset * 35.0)
        candidate_score = self._candidate_score(
            lip_motion=lip_motion,
            gaze_score=gaze_score,
            head_yaw_deg=head_yaw_deg,
            head_pitch_deg=head_pitch_deg,
            direction_deg=direction_deg,
            distance_m=distance_m,
            mouth_occluded=mouth_occluded,
        )

        return {
            "candidate_id": candidate_id,
            "candidate_score": candidate_score,
            "face": face,
            "landmarks": landmarks,
            "frame_width": width,
            "frame_height": height,
            "direction_deg": direction_deg,
            "face_direction_deg": direction_deg,
            "sound_face_match_score": 1.0,
            "head_yaw_deg": head_yaw_deg,
            "head_pitch_deg": head_pitch_deg,
            "gaze_score": gaze_score,
            "gaze_active": gaze_active,
            "distance_m": distance_m,
            "lip_motion": lip_motion,
            "eye_occlusion": eye_occlusion,
            "mouth_occluded": mouth_occluded,
            "side_profile": side_profile,
        }

    def _select_face_mesh_candidate(self, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
        eligible = [candidate for candidate in candidates if candidate["candidate_score"] > 0.0]
        if not eligible:
            return None, len(candidates) > 1
        ranked = sorted(eligible, key=lambda candidate: candidate["candidate_score"], reverse=True)
        if len(ranked) >= 2:
            if self._candidate_has_clear_direct_attention(ranked[0], ranked[1]):
                return ranked[0], False
            lip_gap = ranked[0].get("lip_motion", 0.0) - ranked[1].get("lip_motion", 0.0)
            if lip_gap >= MULTI_PERSON_LIP_DOMINANCE_MARGIN and ranked[0].get("lip_motion", 0.0) >= self._config.min_mouth_motion:
                return ranked[0], False
        if len(ranked) >= 2 and ranked[0]["candidate_score"] - ranked[1]["candidate_score"] < MULTI_PERSON_SELECTION_MARGIN:
            return None, True
        return ranked[0], False

    def _candidate_has_clear_direct_attention(self, candidate: dict[str, Any], runner_up: dict[str, Any]) -> bool:
        lip_motion = candidate.get("lip_motion", 0.0)
        if lip_motion < self._config.min_mouth_motion:
            return False

        gaze_score = _clamp_local(candidate.get("gaze_score", 0.0))
        runner_up_gaze = _clamp_local(runner_up.get("gaze_score", 0.0))
        if gaze_score < GAZE_ENTER_THRESHOLD or gaze_score - runner_up_gaze < 0.20:
            return False

        head_score = _head_facing_score_local(candidate.get("head_yaw_deg"))
        direction_score = _direction_score_local(candidate.get("direction_deg"))
        return max(head_score, direction_score) >= 0.70

    def _human_conversation_score(self, candidates: list[dict[str, Any]], selected: dict[str, Any], ambiguous: bool) -> float:
        if len(candidates) < 2:
            return 0.0
        active_lips = [candidate.get("lip_motion", 0.0) for candidate in candidates if candidate.get("lip_motion", 0.0) >= self._config.min_mouth_motion]
        if not active_lips:
            return 0.0

        gaze_score = _clamp_local(selected.get("gaze_score", 0.0))
        facing_score = _head_facing_score_local(selected.get("head_yaw_deg"))
        direction_score = _direction_score_local(selected.get("direction_deg"))
        low_loona_attention = _clamp_local((0.65 - gaze_score) / 0.65)

        lip_scores = sorted((candidate.get("lip_motion", 0.0) for candidate in candidates), reverse=True)
        second_lip_ratio = lip_scores[1] / max(lip_scores[0], 1e-6) if len(lip_scores) >= 2 else 0.0
        shared_lip_activity = _clamp_local((second_lip_ratio - 0.35) / 0.45) if len(active_lips) >= 2 else 0.0
        off_axis_score = 1.0 - max(facing_score, direction_score)
        ambiguous_score = 0.35 if ambiguous else 0.0

        return _clamp_local(
            (0.65 * low_loona_attention)
            + (0.22 * shared_lip_activity)
            + (0.13 * off_axis_score)
            + ambiguous_score
        )

    def _assign_track_id(self, face: tuple[int, int, int, int], used_track_ids: set[str] | None = None) -> str:
        used_track_ids = used_track_ids if used_track_ids is not None else set()
        center = self._bbox_center(face)
        face_scale = max(float(face[2]), float(face[3]), 1.0)
        best_track_id: str | None = None
        best_distance = float("inf")
        for track_id, track in self._tracks.items():
            if track_id in used_track_ids:
                continue
            distance = float(np.linalg.norm(np.asarray(center) - np.asarray(track["center"])))
            threshold = max(face_scale, float(track.get("scale", face_scale))) * TRACK_MATCH_DISTANCE_RATIO
            if distance <= threshold and distance < best_distance:
                best_track_id = track_id
                best_distance = distance

        if best_track_id is None:
            best_track_id = f"local_user_{self._next_track_index}"
            self._next_track_index += 1

        self._tracks[best_track_id] = {
            "center": center,
            "scale": face_scale,
            "last_seen": self._frame_index,
        }
        used_track_ids.add(best_track_id)
        return best_track_id

    def _drop_stale_tracks(self, active_track_ids: set[str]) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in active_track_ids and self._frame_index - int(track.get("last_seen", self._frame_index)) > TRACK_MAX_STALE_FRAMES
        ]
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)
            self._prev_lip_open_ratios.pop(track_id, None)
            self._mouth_motion_histories.pop(track_id, None)
            self._gaze_histories.pop(track_id, None)
            self._gaze_states.pop(track_id, None)

    def _bbox_center(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        left, top, width, height = bbox
        return (left + (width / 2.0), top + (height / 2.0))

    def _candidate_score(
        self,
        *,
        lip_motion: float,
        gaze_score: float,
        head_yaw_deg: float | None,
        head_pitch_deg: float | None,
        direction_deg: float | None,
        distance_m: float | None,
        mouth_occluded: bool,
    ) -> float:
        if mouth_occluded or lip_motion < self._config.min_mouth_motion:
            return 0.0
        if head_yaw_deg is not None and abs(head_yaw_deg) > MAX_HEAD_YAW_DEG:
            return 0.0
        if head_pitch_deg is not None and abs(head_pitch_deg) > MAX_HEAD_PITCH_DEG:
            return 0.0
        return _clamp_local(
            (lip_motion * 0.45)
            + (gaze_score * 0.22)
            + (_head_facing_score_local(head_yaw_deg) * 0.18)
            + (_direction_score_local(direction_deg) * 0.10)
            + (_distance_score_local(distance_m) * 0.05)
        )

    def _detect_hands(self, mp_image: Any, timestamp_ms: int) -> Any | None:
        if self._hand_landmarker is None:
            return None
        return self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    def _face_matrix_yaw_deg(self, result: Any, face_index: int = 0) -> float | None:
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if not matrices or face_index >= len(matrices):
            return None
        matrix = np.asarray(matrices[face_index], dtype=np.float32)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return None
        rotation = matrix[:3, :3]
        yaw_rad = np.arctan2(rotation[0, 2], rotation[2, 2])
        return float(np.degrees(yaw_rad))

    def _face_matrix_pitch_deg(self, result: Any, face_index: int = 0) -> float | None:
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if not matrices or face_index >= len(matrices):
            return None
        matrix = np.asarray(matrices[face_index], dtype=np.float32)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return None
        rotation = matrix[:3, :3]
        pitch_rad = np.arctan2(-rotation[1, 2], rotation[1, 1])
        return float(np.degrees(pitch_rad))

    def _is_side_profile(self, landmarks, width: int, height: int, head_yaw_deg: float | None) -> bool:  # noqa: ANN001
        if head_yaw_deg is not None and abs(head_yaw_deg) >= 12.0:
            return True
        face_points = self._landmark_points(landmarks, width, height, FACE_OVAL_POINTS)
        face_left, _, face_width, _ = self._points_bbox(face_points)
        nose_tip = landmarks[1]
        nose_x = float(nose_tip.x * width)
        face_center_x = face_left + (face_width / 2.0)
        nose_offset_ratio = abs(nose_x - face_center_x) / max(float(face_width), 1.0)
        return nose_offset_ratio >= 0.11

    def _face_contour_indices(self, side_profile: bool):  # noqa: ANN202
        return FACE_OVAL_POINTS if side_profile else range(478)

    def _landmark_points(self, landmarks, width: int, height: int, indices) -> list[tuple[int, int]]:  # noqa: ANN001
        points = []
        for index in indices:
            landmark = landmarks[index]
            points.append((int(landmark.x * width), int(landmark.y * height)))
        return points

    def _points_bbox(self, points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        left = max(0, min(x_values))
        top = max(0, min(y_values))
        right = max(x_values)
        bottom = max(y_values)
        return (left, top, max(1, right - left), max(1, bottom - top))

    def _estimate_mesh_lip_motion(self, landmarks, width: int, height: int, candidate_id: str = "local_user") -> float:  # noqa: ANN001
        open_ratio = self._mesh_lip_open_ratio(landmarks, width, height)
        previous_ratio = self._prev_lip_open_ratios.get(candidate_id)
        delta = 0.0 if previous_ratio is None else abs(open_ratio - previous_ratio)
        self._prev_lip_open_ratios[candidate_id] = open_ratio
        self._prev_lip_open_ratio = open_ratio
        score = max(0.0, delta - LIP_OPEN_RATIO_DEADZONE) * LIP_OPEN_RATIO_SCORE_SCALE
        history = self._mouth_motion_histories.setdefault(candidate_id, deque(maxlen=5))
        history.append(float(score))
        self._mouth_motion_history = history
        return max(0.0, min(1.0, float(np.mean(history))))

    def _mesh_lip_open_ratio(self, landmarks, width: int, height: int) -> float:  # noqa: ANN001
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        mouth_width_px = max(abs((right_mouth.x - left_mouth.x) * width), 1.0)
        gaps = [abs((landmarks[lower].y - landmarks[upper].y) * height) / mouth_width_px for upper, lower in LIP_OPENING_PAIRS]
        return float(np.median(gaps)) if gaps else 0.0

    def _estimate_mesh_gaze_score(
        self,
        landmarks,
        width: int,
        height: int,
        face_center_score: float,
        eye_occlusion: dict[str, bool] | None = None,
    ) -> float:  # noqa: ANN001
        if len(landmarks) < 478:
            return max(0.0, min(1.0, face_center_score * 0.65))
        eye_occlusion = eye_occlusion or {"left": False, "right": False}

        left_eye = self._landmark_points(landmarks, width, height, LEFT_EYE_POINTS)
        right_eye = self._landmark_points(landmarks, width, height, RIGHT_EYE_POINTS)
        left_iris = self._landmark_points(landmarks, width, height, range(468, 473))
        right_iris = self._landmark_points(landmarks, width, height, range(473, 478))
        eye_scores = []
        openness_scores = []
        if not eye_occlusion.get("left", False):
            eye_scores.append(self._iris_center_score(left_eye, left_iris))
            openness_scores.append(self._eye_openness_score(left_eye))
        if not eye_occlusion.get("right", False):
            eye_scores.append(self._iris_center_score(right_eye, right_iris))
            openness_scores.append(self._eye_openness_score(right_eye))
        if not eye_scores or not openness_scores:
            return 0.0
        openness_score = sum(openness_scores) / len(openness_scores)
        if openness_score < 0.45:
            return 0.0
        iris_score = sum(eye_scores) / len(eye_scores)
        return max(0.0, min(1.0, (iris_score * 0.62) + (openness_score * 0.18) + (face_center_score * 0.20)))

    def _stabilized_gaze_score(self, candidate_id: str, gaze_score: float, eye_occlusion: dict[str, bool]) -> float:
        if eye_occlusion.get("left", False) and eye_occlusion.get("right", False):
            self._gaze_histories.pop(candidate_id, None)
            self._gaze_states.pop(candidate_id, None)
            return 0.0
        history = self._gaze_histories.setdefault(candidate_id, deque(maxlen=4))
        history.append(_clamp_local(gaze_score))
        return float(sum(history) / len(history))

    def _stable_gaze_state(self, candidate_id: str, gaze_score: float) -> bool:
        previous = self._gaze_states.get(candidate_id, False)
        if previous:
            active = gaze_score >= GAZE_EXIT_THRESHOLD
        else:
            active = gaze_score >= GAZE_ENTER_THRESHOLD
        self._gaze_states[candidate_id] = active
        return active

    def _eye_occlusion_state(
        self,
        landmarks,
        rgb: np.ndarray,
        hand_result: Any | None,
        width: int,
        height: int,
    ) -> dict[str, bool]:  # noqa: ANN001
        return {
            "left": self._eye_is_occluded(landmarks, rgb, hand_result, width, height, LEFT_EYE_POINTS, range(468, 473)),
            "right": self._eye_is_occluded(landmarks, rgb, hand_result, width, height, RIGHT_EYE_POINTS, range(473, 478)),
        }

    def _eye_is_occluded(
        self,
        landmarks,
        rgb: np.ndarray,
        hand_result: Any | None,
        width: int,
        height: int,
        eye_indices: list[int],
        iris_indices,
    ) -> bool:  # noqa: ANN001
        eye_points = self._landmark_points(landmarks, width, height, eye_indices)
        if self._eye_is_occluded_by_hand(eye_points, hand_result, width, height):
            return True
        if self._eye_openness_score(eye_points) < 0.45:
            return False
        iris_points = self._landmark_points(landmarks, width, height, iris_indices)
        return self._eye_visual_evidence_score(rgb, eye_points, iris_points) < EYE_OCCLUSION_EVIDENCE_THRESHOLD

    def _mouth_is_occluded(
        self,
        landmarks,
        rgb: np.ndarray,
        hand_result: Any | None,
        width: int,
        height: int,
    ) -> bool:  # noqa: ANN001
        mouth_points = self._landmark_points(landmarks, width, height, OUTER_LIP_POINTS + INNER_LIP_POINTS)
        if self._points_are_occluded_by_hand(
            mouth_points,
            hand_result,
            width,
            height,
            padding_ratio=0.65,
            min_region_overlap_ratio=0.18,
        ):
            return True
        return self._mouth_visual_evidence_score(rgb, mouth_points) < MOUTH_OCCLUSION_EVIDENCE_THRESHOLD

    def _eyes_are_occluded_by_hand(self, landmarks, hand_result: Any | None, width: int, height: int) -> bool:  # noqa: ANN001
        eye_points = self._landmark_points(landmarks, width, height, LEFT_EYE_POINTS + RIGHT_EYE_POINTS)
        return self._eye_is_occluded_by_hand(eye_points, hand_result, width, height)

    def _eye_is_occluded_by_hand(
        self,
        eye_points: list[tuple[int, int]],
        hand_result: Any | None,
        width: int,
        height: int,
    ) -> bool:
        hand_landmarks = getattr(hand_result, "hand_landmarks", None)
        if not hand_landmarks:
            return False
        return self._points_are_occluded_by_hand(eye_points, hand_result, width, height, padding_ratio=0.55)

    def _points_are_occluded_by_hand(
        self,
        points: list[tuple[int, int]],
        hand_result: Any | None,
        width: int,
        height: int,
        *,
        padding_ratio: float,
        min_region_overlap_ratio: float = 0.0,
    ) -> bool:
        hand_landmarks = getattr(hand_result, "hand_landmarks", None)
        if not hand_landmarks:
            return False
        region_box = self._expanded_bbox(points, width, height, padding_ratio=padding_ratio)
        hand_points = []
        for hand in hand_landmarks:
            current_hand_points = [(int(point.x * width), int(point.y * height)) for point in hand]
            hand_points.extend(current_hand_points)
            if min_region_overlap_ratio > 0.0 and self._bbox_overlap_ratio(
                region_box,
                self._expanded_bbox(current_hand_points, width, height, padding_ratio=0.12),
            ) >= min_region_overlap_ratio:
                return True
        return any(self._point_inside_bbox(point, region_box) for point in hand_points)

    def _eye_visual_evidence_score(
        self,
        rgb: np.ndarray,
        eye_points: list[tuple[int, int]],
        iris_points: list[tuple[int, int]],
    ) -> float:
        eye_box = self._expanded_bbox(eye_points, rgb.shape[1], rgb.shape[0], padding_ratio=0.30)
        iris_box = self._expanded_bbox(iris_points, rgb.shape[1], rgb.shape[0], padding_ratio=0.20)
        eye_crop = self._crop_bbox(rgb, eye_box)
        iris_crop = self._crop_bbox(rgb, iris_box)
        if eye_crop.size == 0 or iris_crop.size == 0:
            return 0.0
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY)
        iris_gray = cv2.cvtColor(iris_crop, cv2.COLOR_RGB2GRAY)
        contrast = max(0.0, (float(np.mean(eye_gray)) - float(np.mean(iris_gray))) / 80.0)
        texture = min(1.0, float(np.std(eye_gray)) / 35.0)
        return max(0.0, min(1.0, (contrast * 0.65) + (texture * 0.35)))

    def _mouth_visual_evidence_score(self, rgb: np.ndarray, mouth_points: list[tuple[int, int]]) -> float:
        mouth_box = self._expanded_bbox(mouth_points, rgb.shape[1], rgb.shape[0], padding_ratio=0.25)
        mouth_crop = self._crop_bbox(rgb, mouth_box)
        if mouth_crop.size == 0:
            return 0.0
        mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY)
        texture = min(1.0, float(np.std(mouth_gray)) / 32.0)
        edges = cv2.Canny(mouth_gray, 40, 120)
        edge_density = min(1.0, float(np.count_nonzero(edges)) / max(float(edges.size) * 0.18, 1.0))
        return max(0.0, min(1.0, (texture * 0.55) + (edge_density * 0.45)))

    def _crop_bbox(self, rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        left, top, right, bottom = bbox
        return rgb[top : bottom + 1, left : right + 1]

    def _expanded_bbox(
        self,
        points: list[tuple[int, int]],
        width: int,
        height: int,
        *,
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        left, top, box_width, box_height = self._points_bbox(points)
        padding_x = int(box_width * padding_ratio)
        padding_y = int(box_height * padding_ratio)
        expanded_left = max(0, left - padding_x)
        expanded_top = max(0, top - padding_y)
        expanded_right = min(width - 1, left + box_width + padding_x)
        expanded_bottom = min(height - 1, top + box_height + padding_y)
        return (expanded_left, expanded_top, expanded_right, expanded_bottom)

    def _point_inside_bbox(self, point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
        x, y = point
        left, top, right, bottom = bbox
        return left <= x <= right and top <= y <= bottom

    def _bbox_overlap_ratio(self, region: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> float:
        left = max(region[0], other[0])
        top = max(region[1], other[1])
        right = min(region[2], other[2])
        bottom = min(region[3], other[3])
        if right < left or bottom < top:
            return 0.0
        overlap_area = float((right - left + 1) * (bottom - top + 1))
        region_area = float((region[2] - region[0] + 1) * (region[3] - region[1] + 1))
        return overlap_area / max(region_area, 1.0)

    def _iris_center_score(self, eye_points: list[tuple[int, int]], iris_points: list[tuple[int, int]]) -> float:
        if not eye_points or not iris_points:
            return 0.0
        eye_array = np.asarray(eye_points, dtype=np.float32)
        iris_array = np.asarray(iris_points, dtype=np.float32)
        eye_left = float(np.min(eye_array[:, 0]))
        eye_right = float(np.max(eye_array[:, 0]))
        eye_top = float(np.min(eye_array[:, 1]))
        eye_bottom = float(np.max(eye_array[:, 1]))
        eye_width = max(eye_right - eye_left, 1.0)
        eye_height = max(eye_bottom - eye_top, 1.0)
        iris_center = iris_array.mean(axis=0)
        horizontal_offset = abs(((iris_center[0] - eye_left) / eye_width) - 0.5) / 0.5
        vertical_offset = abs(((iris_center[1] - eye_top) / eye_height) - 0.5) / 0.5
        return max(0.0, min(1.0, 1.0 - ((horizontal_offset * 0.75) + (vertical_offset * 0.25))))

    def _eye_openness_score(self, eye_points: list[tuple[int, int]]) -> float:
        if not eye_points:
            return 0.0
        eye_array = np.asarray(eye_points, dtype=np.float32)
        eye_width = max(float(np.max(eye_array[:, 0]) - np.min(eye_array[:, 0])), 1.0)
        eye_height = float(np.max(eye_array[:, 1]) - np.min(eye_array[:, 1]))
        openness_ratio = eye_height / eye_width
        return max(0.0, min(1.0, (openness_ratio - 0.10) / 0.12))

    def _estimate_mouth_motion(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> float:
        x, y, w, h = face
        mouth_y = y + int(h * 0.58)
        mouth_h = int(h * 0.28)
        mouth_x = x + int(w * 0.20)
        mouth_w = int(w * 0.60)
        mouth = gray[mouth_y : mouth_y + mouth_h, mouth_x : mouth_x + mouth_w]
        if mouth.size == 0:
            return 0.0
        mouth = cv2.resize(mouth, (96, 48))
        if self._prev_mouth_gray is None:
            self._prev_mouth_gray = mouth
            return 0.0
        diff = cv2.absdiff(mouth, self._prev_mouth_gray)
        self._prev_mouth_gray = mouth
        motion = float(np.mean(diff) / 255.0)
        self._mouth_motion_history.append(motion)
        return max(self._mouth_motion_history) if self._mouth_motion_history else motion

    def _detect_eyes(self, gray: np.ndarray, face: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
        x, y, w, h = face
        upper_face = gray[y : y + int(h * 0.55), x : x + w]
        if upper_face.size == 0:
            return []
        eyes = self._eyeglasses_eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(18, 18),
        )
        if len(eyes) == 0:
            eyes = self._eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=7, minSize=(18, 18))
        eye_regions = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
        eye_regions = self._filter_eye_regions(face, eye_regions)
        eye_regions.sort(key=lambda item: item[2] * item[3], reverse=True)
        return eye_regions[:2]

    def _estimate_gaze_score(self, face_center_score: float, eyes: list[tuple[int, int, int, int]]) -> float:
        if not eyes:
            return max(0.0, min(0.75, face_center_score * 0.75))
        eye_visibility_score = 1.0 if len(eyes) >= 2 else 0.65
        return max(0.0, min(1.0, (face_center_score * 0.75) + (eye_visibility_score * 0.25)))

    def _filter_eye_regions(
        self,
        face: tuple[int, int, int, int],
        eyes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        x, y, w, h = face
        filtered = []
        for eye_x, eye_y, eye_w, eye_h in eyes:
            center_x = eye_x + (eye_w / 2.0)
            center_y = eye_y + (eye_h / 2.0)
            in_eye_band = y + (h * 0.16) <= center_y <= y + (h * 0.52)
            inside_face = x <= center_x <= x + w
            reasonable_size = eye_w <= w * 0.45 and eye_h <= h * 0.25
            if in_eye_band and inside_face and reasonable_size:
                filtered.append((eye_x, eye_y, eye_w, eye_h))
        return filtered

    def _largest_face(self, faces) -> tuple[int, int, int, int] | None:  # noqa: ANN001
        if len(faces) == 0:
            self._prev_mouth_gray = None
            self._prev_lip_open_ratio = None
            self._mouth_motion_history.clear()
            self._prev_lip_open_ratios.clear()
            self._mouth_motion_histories.clear()
            self._gaze_histories.clear()
            self._gaze_states.clear()
            self._tracks.clear()
            return None
        return tuple(max(faces, key=lambda item: item[2] * item[3]))

    def _draw_face_mesh_overlays(self, rgb: np.ndarray, mesh_result: dict[str, Any]) -> None:
        landmarks = mesh_result["landmarks"]
        width = mesh_result["frame_width"]
        height = mesh_result["frame_height"]
        lip_motion = mesh_result["lip_motion"]
        gaze_score = mesh_result["gaze_score"]
        head_pitch_deg = mesh_result.get("head_pitch_deg")
        face = mesh_result["face"]
        eye_occlusion = mesh_result.get("eye_occlusion", {"left": False, "right": False})
        mouth_occluded = bool(mesh_result.get("mouth_occluded", False))
        head_angle_valid = self._head_angle_is_valid(mesh_result["head_yaw_deg"])
        face_color = HUD_GREEN if head_angle_valid else HUD_RED
        if mesh_result.get("multi_person_ambiguous"):
            face_color = HUD_RED
        mouth_color = HUD_GREEN if head_angle_valid and self._lip_is_moving(lip_motion) else HUD_RED
        gaze_active = bool(mesh_result.get("gaze_active", self._eyes_are_gazing(gaze_score)))
        gaze_color = HUD_GREEN if head_angle_valid and gaze_active else HUD_RED

        side_profile = bool(mesh_result.get("side_profile", False))
        for candidate in mesh_result.get("candidates", []):
            if candidate.get("candidate_id") != mesh_result.get("candidate_id"):
                self._draw_secondary_face_mesh_overlay(rgb, candidate)

        face_landmark_indices = range(len(landmarks))
        face_reference_points = self._landmark_points(landmarks, width, height, FACE_OVAL_POINTS) if side_profile else None
        face_points = self._outer_face_hull(
            self._landmark_points(landmarks, width, height, face_landmark_indices),
            width,
            height,
            side_profile=side_profile,
            reference_points=face_reference_points,
        )
        outer_lip_points = self._landmark_points(landmarks, width, height, OUTER_LIP_POINTS)
        inner_lip_points = self._landmark_points(landmarks, width, height, INNER_LIP_POINTS)
        left_eye_points = self._landmark_points(landmarks, width, height, LEFT_EYE_POINTS)
        right_eye_points = self._landmark_points(landmarks, width, height, RIGHT_EYE_POINTS)

        self._draw_polyline(rgb, face_points, face_color, closed=True, dashed=True, node_step=4, node_radius=2)
        if not mouth_occluded:
            self._draw_polyline(rgb, outer_lip_points, mouth_color, closed=True, dashed=False, node_step=4, node_radius=1)
            self._draw_polyline(rgb, inner_lip_points, mouth_color, closed=True, dashed=True, node_step=5, node_radius=1)
        if not eye_occlusion.get("left", False):
            self._draw_polyline(rgb, left_eye_points, gaze_color, closed=True, dashed=False, node_step=4, node_radius=1)
        if not eye_occlusion.get("right", False):
            self._draw_polyline(rgb, right_eye_points, gaze_color, closed=True, dashed=False, node_step=4, node_radius=1)

        cv2.putText(
            rgb,
            f"mesh  lip {lip_motion:.3f}  gaze {gaze_score:.2f}  yaw {(mesh_result['head_yaw_deg'] or 0.0):.0f}  pitch {(head_pitch_deg or 0.0):.0f}  {'side' if side_profile else 'front'}",
            (face[0], max(20, face[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HUD_TEXT,
            1,
            cv2.LINE_AA,
        )

    def _draw_secondary_face_mesh_overlay(self, rgb: np.ndarray, candidate: dict[str, Any]) -> None:
        landmarks = candidate["landmarks"]
        width = candidate["frame_width"]
        height = candidate["frame_height"]
        side_profile = bool(candidate.get("side_profile", False))
        reference_points = self._landmark_points(landmarks, width, height, FACE_OVAL_POINTS) if side_profile else None
        face_points = self._outer_face_hull(
            self._landmark_points(landmarks, width, height, range(len(landmarks))),
            width,
            height,
            side_profile=side_profile,
            reference_points=reference_points,
        )
        self._draw_polyline(rgb, face_points, HUD_DIM, closed=True, dashed=True, node_step=6, node_radius=1)

    def _outer_face_hull(
        self,
        points: list[tuple[int, int]],
        width: int,
        height: int,
        *,
        side_profile: bool = False,
        reference_points: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        if len(points) < 3:
            return points
        hull = cv2.convexHull(np.asarray(points, dtype=np.int32)).reshape(-1, 2).astype(np.float32)
        if side_profile and reference_points:
            hull = self._limit_side_profile_hull(hull, np.asarray(reference_points, dtype=np.float32))
        center = hull.mean(axis=0)
        face_width = max(float(np.max(hull[:, 0]) - np.min(hull[:, 0])), 1.0)
        face_height = max(float(np.max(hull[:, 1]) - np.min(hull[:, 1])), 1.0)
        if side_profile:
            scale = 1.014
        else:
            scale = 1.015 + min(0.035, abs(face_width - face_height) / max(face_width, face_height) * 0.035)
        expanded = center + ((hull - center) * scale)
        expanded = self._extend_forehead_contour(expanded, face_height, side_profile=side_profile)
        expanded = self._smooth_closed_contour(expanded, iterations=1 if side_profile else 2)
        if side_profile:
            expanded = self._restore_contour_bounds(expanded, hull, padding=face_width * 0.012)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
        epsilon = max(0.8, cv2.arcLength(expanded.astype(np.float32), True) * 0.003)
        simplified = cv2.approxPolyDP(expanded.astype(np.float32), epsilon, True).reshape(-1, 2)
        return [(int(x), int(y)) for x, y in simplified]

    def _limit_side_profile_hull(self, hull: np.ndarray, reference: np.ndarray) -> np.ndarray:
        reference_left = float(np.min(reference[:, 0]))
        reference_right = float(np.max(reference[:, 0]))
        reference_top = float(np.min(reference[:, 1]))
        reference_bottom = float(np.max(reference[:, 1]))
        reference_width = max(reference_right - reference_left, 1.0)
        reference_height = max(reference_bottom - reference_top, 1.0)
        left_limit = reference_left - (reference_width * 0.13)
        right_limit = reference_right + (reference_width * 0.13)
        top_limit = reference_top - (reference_height * 0.09)
        bottom_limit = reference_bottom + (reference_height * 0.020)
        limited = hull.copy()
        limited[:, 0] = np.clip(limited[:, 0], left_limit, right_limit)
        limited[:, 1] = np.clip(limited[:, 1], top_limit, bottom_limit)
        return cv2.convexHull(limited.astype(np.float32)).reshape(-1, 2)

    def _extend_forehead_contour(self, hull: np.ndarray, face_height: float, *, side_profile: bool = False) -> np.ndarray:
        adjusted = hull.copy()
        top = float(np.min(adjusted[:, 1]))
        bottom = float(np.max(adjusted[:, 1]))
        left = float(np.min(adjusted[:, 0]))
        right = float(np.max(adjusted[:, 0]))
        center_x = (left + right) / 2.0
        half_width = max((right - left) / 2.0, 1.0)
        upper_band = max((bottom - top) * (0.30 if side_profile else 0.34), 1.0)
        max_lift = face_height * (0.055 if side_profile else 0.12)
        for index, point in enumerate(adjusted):
            upper_position = (point[1] - top) / upper_band
            if upper_position <= 1.0:
                horizontal_position = abs(point[0] - center_x) / half_width
                exponent = 2.7 if side_profile else 2.4
                center_weight = max(0.0, 1.0 - (horizontal_position**exponent))
                vertical_base = max(0.0, 1.0 - upper_position)
                vertical_weight = vertical_base ** (1.45 if side_profile else 1.35)
                arc_y = top - (max_lift * center_weight) + (face_height * 0.025 * (horizontal_position**2))
                lift_factor = 0.28 if side_profile else 0.35
                lifted_y = point[1] - (max_lift * vertical_weight * center_weight * lift_factor)
                min_center_weight = 0.35 if side_profile else 0.18
                adjusted[index, 1] = min(lifted_y, arc_y) if center_weight > min_center_weight else lifted_y
        return adjusted

    def _restore_contour_bounds(self, contour: np.ndarray, reference: np.ndarray, padding: float) -> np.ndarray:
        restored = contour.copy()
        reference_left = float(np.min(reference[:, 0])) - padding
        reference_right = float(np.max(reference[:, 0])) + padding
        reference_top = float(np.min(reference[:, 1])) - (padding * 0.8)
        current_left = float(np.min(restored[:, 0]))
        current_right = float(np.max(restored[:, 0]))
        current_top = float(np.min(restored[:, 1]))
        current_center_x = (current_left + current_right) / 2.0
        target_center_x = (reference_left + reference_right) / 2.0
        current_width = max(current_right - current_left, 1.0)
        target_width = max(reference_right - reference_left, current_width)
        width_scale = target_width / current_width
        restored[:, 0] = target_center_x + ((restored[:, 0] - current_center_x) * width_scale)
        if current_top > reference_top:
            restored[:, 1] -= current_top - reference_top
        return restored

    def _smooth_closed_contour(self, contour: np.ndarray, iterations: int) -> np.ndarray:
        smoothed = contour.astype(np.float32)
        if len(smoothed) < 3:
            return smoothed
        for _ in range(iterations):
            refined = []
            for index, point in enumerate(smoothed):
                next_point = smoothed[(index + 1) % len(smoothed)]
                refined.append((point * 0.75) + (next_point * 0.25))
                refined.append((point * 0.25) + (next_point * 0.75))
            smoothed = np.asarray(refined, dtype=np.float32)
        return smoothed

    def _draw_polyline(
        self,
        rgb: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        *,
        closed: bool,
        dashed: bool,
        node_step: int = 0,
        node_radius: int = 2,
    ) -> None:
        if len(points) < 2:
            return
        draw_points = points + [points[0]] if closed else points
        for index, (start, end) in enumerate(zip(draw_points, draw_points[1:])):
            if dashed:
                self._draw_dashed_line(rgb, start, end, color, dash=4.0, gap=5.0, phase=float(index % 2) * 1.5)
            else:
                cv2.line(rgb, start, end, self._dim_color(color, 0.52), 1, lineType=cv2.LINE_AA)
        if node_step > 0:
            node_points = points[::node_step]
            self._draw_hud_nodes(rgb, node_points, color, radius=node_radius)

    def _draw_dashed_line(
        self,
        rgb: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        *,
        dash: float = 5.0,
        gap: float = 4.0,
        phase: float = 0.0,
    ) -> None:
        start_array = np.asarray(start, dtype=np.float32)
        end_array = np.asarray(end, dtype=np.float32)
        vector = end_array - start_array
        length = float(np.linalg.norm(vector))
        if length < 2.0:
            return
        direction = vector / length
        position = phase
        while position < length:
            segment_start = start_array + direction * position
            segment_end = start_array + direction * min(position + dash, length)
            cv2.line(
                rgb,
                tuple(segment_start.astype(int)),
                tuple(segment_end.astype(int)),
                self._dim_color(color, 0.52),
                1,
                lineType=cv2.LINE_AA,
            )
            position += dash + gap

    def _draw_hud_nodes(
        self,
        rgb: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        *,
        radius: int,
    ) -> None:
        for point in points:
            cv2.circle(rgb, point, radius, self._dim_color(color, 0.62), -1, lineType=cv2.LINE_AA)

    def _draw_hud_ellipse(
        self,
        rgb: np.ndarray,
        center: tuple[int, int],
        axes: tuple[int, int],
        color: tuple[int, int, int],
        *,
        thickness: int = 2,
        node_radius: int = 2,
    ) -> None:
        cv2.ellipse(rgb, center, axes, 0, 0, 360, self._dim_color(color, 0.75), thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(rgb, center, axes, 0, 0, 360, self._bright_color(color, 0.25), 1, lineType=cv2.LINE_AA)
        for angle in range(0, 360, 45):
            point_x = int(center[0] + axes[0] * np.cos(np.deg2rad(angle)))
            point_y = int(center[1] + axes[1] * np.sin(np.deg2rad(angle)))
            cv2.circle(rgb, (point_x, point_y), node_radius, self._dim_color(color, 0.65), -1, lineType=cv2.LINE_AA)

    def _bright_color(self, color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        return tuple(min(255, int(channel + ((255 - channel) * amount))) for channel in color)

    def _dim_color(self, color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        return tuple(max(0, int(channel * amount)) for channel in color)

    def _draw_detection_overlays(
        self,
        rgb: np.ndarray,
        face: tuple[int, int, int, int],
        lip_motion: float,
        eyes: list[tuple[int, int, int, int]],
        gaze_score: float,
    ) -> None:
        x, y, w, h = face
        head_yaw_deg = ((x + (w / 2.0)) - (rgb.shape[1] / 2.0)) / max(rgb.shape[1] / 2.0, 1.0) * 35.0
        head_angle_valid = self._head_angle_is_valid(head_yaw_deg)
        face_color = HUD_GREEN if head_angle_valid else HUD_RED
        mouth_color = HUD_GREEN if head_angle_valid and self._lip_is_moving(lip_motion) else HUD_RED
        gaze_color = HUD_GREEN if head_angle_valid and self._eyes_are_gazing(gaze_score) else HUD_RED

        self._draw_hud_ellipse(
            rgb,
            (x + w // 2, y + h // 2),
            (max(w // 2, 1), max(int(h * 0.56), 1)),
            face_color,
        )

        mouth_y = y + int(h * 0.58)
        mouth_h = int(h * 0.28)
        mouth_x = x + int(w * 0.20)
        mouth_w = int(w * 0.60)
        self._draw_hud_ellipse(
            rgb,
            (mouth_x + mouth_w // 2, mouth_y + mouth_h // 2),
            (max(mouth_w // 2, 1), max(mouth_h // 3, 1)),
            mouth_color,
            thickness=1,
            node_radius=1,
        )

        if eyes:
            for eye_x, eye_y, eye_w, eye_h in eyes:
                self._draw_hud_ellipse(
                    rgb,
                    (eye_x + eye_w // 2, eye_y + eye_h // 2),
                    (max(eye_w // 2, 1), max(eye_h // 3, 1)),
                    gaze_color,
                    thickness=1,
                    node_radius=1,
                )
        else:
            left_eye_center = (x + int(w * 0.35), y + int(h * 0.34))
            right_eye_center = (x + int(w * 0.65), y + int(h * 0.34))
            eye_axes = (max(int(w * 0.11), 1), max(int(h * 0.04), 1))
            self._draw_hud_ellipse(rgb, left_eye_center, eye_axes, gaze_color, thickness=1, node_radius=1)
            self._draw_hud_ellipse(rgb, right_eye_center, eye_axes, gaze_color, thickness=1, node_radius=1)

        cv2.putText(
            rgb,
            f"face ok  lip {lip_motion:.3f}  gaze {gaze_score:.2f}",
            (x,
             max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HUD_TEXT,
            1,
            cv2.LINE_AA,
        )

    def _lip_is_moving(self, lip_motion: float) -> bool:
        return lip_motion >= self._config.min_mouth_motion

    def _eyes_are_gazing(self, gaze_score: float) -> bool:
        return gaze_score >= GAZE_ENTER_THRESHOLD

    def _head_angle_is_valid(self, head_yaw_deg: float | None) -> bool:
        return head_yaw_deg is not None and abs(head_yaw_deg) <= MAX_HEAD_YAW_DEG

    def _draw_missing_face(self, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        center = (width // 2, height // 2)
        axes = (max(width // 7, 1), max(height // 4, 1))
        self._draw_hud_ellipse(rgb, center, axes, HUD_RED)

    def _to_qimage(self, rgb: np.ndarray) -> QImage:
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
