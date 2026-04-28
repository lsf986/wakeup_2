from types import SimpleNamespace

import numpy as np
import pytest

from loona_wakeup.adapters.local_camera_mic_adapter import (
    EYE_OCCLUSION_EVIDENCE_THRESHOLD,
    FACE_OVAL_POINTS,
    LEFT_EYE_POINTS,
    MOUTH_OCCLUSION_EVIDENCE_THRESHOLD,
    OUTER_LIP_POINTS,
    INNER_LIP_POINTS,
    RIGHT_EYE_POINTS,
    LocalCameraMicAdapter,
)
from loona_wakeup.models import LocalInputConfig


def _landmarks() -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]


def _set_eye_boxes(landmarks: list[SimpleNamespace]) -> None:
    for offset, index in enumerate(LEFT_EYE_POINTS):
        landmarks[index] = SimpleNamespace(x=0.38 + ((offset % 4) * 0.025), y=0.45 + ((offset % 2) * 0.04))
    for offset, index in enumerate(RIGHT_EYE_POINTS):
        landmarks[index] = SimpleNamespace(x=0.56 + ((offset % 4) * 0.025), y=0.45 + ((offset % 2) * 0.04))


def _set_closed_eye_boxes(landmarks: list[SimpleNamespace]) -> None:
    for offset, index in enumerate(LEFT_EYE_POINTS):
        landmarks[index] = SimpleNamespace(x=0.38 + ((offset % 4) * 0.025), y=0.47 + ((offset % 2) * 0.004))
    for offset, index in enumerate(RIGHT_EYE_POINTS):
        landmarks[index] = SimpleNamespace(x=0.56 + ((offset % 4) * 0.025), y=0.47 + ((offset % 2) * 0.004))


def _set_face_oval_box(landmarks: list[SimpleNamespace]) -> None:
    for offset, index in enumerate(FACE_OVAL_POINTS):
        x = 0.35 + ((offset % 6) * 0.05)
        y = 0.25 + ((offset // 6) * 0.08)
        landmarks[index] = SimpleNamespace(x=x, y=y)


def _set_lip_boxes(landmarks: list[SimpleNamespace]) -> None:
    for offset, index in enumerate(OUTER_LIP_POINTS):
        landmarks[index] = SimpleNamespace(x=0.42 + ((offset % 5) * 0.035), y=0.62 + ((offset // 5) * 0.018))
    for offset, index in enumerate(INNER_LIP_POINTS):
        landmarks[index] = SimpleNamespace(x=0.45 + ((offset % 5) * 0.025), y=0.635 + ((offset // 5) * 0.010))


def test_speech_score_increases_above_noise_floor() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_audio_energy=0.01))
    adapter._noise_floor = 0.001
    adapter._audio_energy = 0.02
    assert adapter._speech_score() > 0.9


def test_speech_score_stays_low_near_noise_floor() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_audio_energy=0.01))
    adapter._noise_floor = 0.01
    adapter._audio_energy = 0.011
    assert adapter._speech_score() == 0.0


def test_speech_quality_keeps_voiced_audio_high() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(audio_sample_rate=16000))
    sample_rate = 16000
    t = np.arange(1024, dtype=np.float32) / sample_rate
    voiced = (0.025 * np.sin(2 * np.pi * 180 * t)) + (0.010 * np.sin(2 * np.pi * 360 * t))
    assert adapter._speech_quality_score(voiced.astype(np.float32), sample_rate) > 0.75


def test_speech_quality_reduces_breath_noise() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(audio_sample_rate=16000))
    rng = np.random.default_rng(7)
    breath_noise = rng.normal(0.0, 0.025, 1024).astype(np.float32)
    assert adapter._speech_quality_score(breath_noise, 16000) < 0.45


def test_speech_score_rejects_loud_breath_noise() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_audio_energy=0.01, audio_sample_rate=16000))
    rng = np.random.default_rng(11)
    breath_noise = rng.normal(0.0, 0.03, 1024).astype(np.float32)
    adapter._noise_floor = 0.001
    adapter._audio_energy = float(np.sqrt(np.mean(np.square(breath_noise))))
    adapter._audio_speech_quality = adapter._speech_quality_score(breath_noise, 16000)
    assert adapter._speech_score() < 0.45


def test_gaze_score_falls_back_to_centered_face_when_eyes_are_hidden() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    assert adapter._estimate_gaze_score(0.95, []) > 0.7


def test_gaze_score_stays_low_for_off_center_face_without_eyes() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    assert adapter._estimate_gaze_score(0.35, []) < 0.55


def test_gaze_score_improves_with_two_visible_eyes() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    eyes = [(10, 10, 20, 12), (40, 10, 20, 12)]
    assert adapter._estimate_gaze_score(0.95, eyes) > 0.9


def test_filter_eye_regions_removes_lower_face_false_positives() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    face = (100, 100, 200, 220)
    eyes = [(140, 150, 40, 24), (180, 250, 80, 70)]
    assert adapter._filter_eye_regions(face, eyes) == [(140, 150, 40, 24)]


def test_points_bbox_wraps_face_mesh_landmarks() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    points = [(10, 20), (50, 80), (20, 40)]
    assert adapter._points_bbox(points) == (10, 20, 40, 60)


def test_outer_face_hull_removes_inward_face_contour_dent() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    points = [(20, 20), (80, 20), (80, 80), (50, 45), (20, 80)]
    hull = adapter._outer_face_hull(points, width=100, height=100)
    assert (50, 45) not in hull
    assert len(hull) > 4


def test_outer_face_hull_smooths_forehead_curve() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    points = [(20, 24), (42, 20), (52, 28), (66, 19), (88, 25), (85, 88), (18, 88)]
    hull = adapter._outer_face_hull(points, width=120, height=120)
    upper_points = [point for point in hull if point[1] <= 32]
    assert len(upper_points) >= 4
    y_steps = [abs(current[1] - following[1]) for current, following in zip(upper_points, upper_points[1:])]
    assert max(y_steps) <= 8


def test_side_profile_face_hull_keeps_expansion_conservative() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    points = [(25, 24), (48, 18), (76, 24), (83, 48), (75, 86), (30, 88), (18, 54)]
    hull = adapter._outer_face_hull(points, width=120, height=120, side_profile=True)
    assert min(point[0] for point in hull) >= 16
    assert max(point[0] for point in hull) <= 85
    assert min(point[1] for point in hull) >= 14


def test_side_profile_face_hull_restores_smoothing_shrinkage() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    points = [(25, 24), (48, 18), (76, 24), (83, 48), (75, 86), (30, 88), (18, 54)]
    hull = adapter._outer_face_hull(points, width=120, height=120, side_profile=True)
    assert min(point[0] for point in hull) <= 18
    assert max(point[0] for point in hull) >= 83
    assert min(point[1] for point in hull) <= 15


def test_side_profile_uses_outer_points_without_unbounded_expansion() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    reference = [(28, 24), (50, 18), (76, 24), (78, 52), (70, 86), (34, 88), (24, 56)]
    all_points = reference + [(16, 50), (86, 46), (54, 12), (94, 44), (10, 52)]
    hull = adapter._outer_face_hull(
        all_points,
        width=120,
        height=120,
        side_profile=True,
        reference_points=reference,
    )
    assert min(point[0] for point in hull) <= 16
    assert max(point[0] for point in hull) <= 89
    assert min(point[1] for point in hull) <= 13


def test_side_profile_detection_uses_nose_offset_when_yaw_is_small() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_face_oval_box(landmarks)
    landmarks[1] = SimpleNamespace(x=0.68, y=0.45)
    assert adapter._is_side_profile(landmarks, width=200, height=200, head_yaw_deg=5.0) is True


def test_front_face_detection_stays_front_when_nose_is_centered() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_face_oval_box(landmarks)
    landmarks[1] = SimpleNamespace(x=0.48, y=0.45)
    assert adapter._is_side_profile(landmarks, width=200, height=200, head_yaw_deg=5.0) is False


def test_side_profile_forehead_lift_is_smaller_than_front_facing_lift() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    contour = np.array([(20, 24), (55, 20), (90, 24), (82, 86), (18, 86)], dtype="float32")
    front = adapter._extend_forehead_contour(contour, face_height=66, side_profile=False)
    side = adapter._extend_forehead_contour(contour, face_height=66, side_profile=True)
    original_top = float(np.min(contour[:, 1]))
    front_lift = original_top - float(np.min(front[:, 1]))
    side_lift = original_top - float(np.min(side[:, 1]))
    assert side_lift < front_lift * 0.5


def test_forehead_contour_extends_upward_without_moving_chin() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    hull = adapter._extend_forehead_contour(
        np.array([(50, 20), (80, 80), (20, 80)], dtype="float32"),
        face_height=60,
    )
    assert hull[0][1] < 20
    assert hull[1][1] == 80
    assert hull[2][1] == 80


def test_forehead_contour_does_not_lift_side_profile_edge_too_much() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    hull = adapter._extend_forehead_contour(
        np.array([(20, 24), (55, 20), (90, 24), (82, 86), (18, 86)], dtype="float32"),
        face_height=66,
    )
    assert hull[0][1] > 22
    assert hull[2][1] > 22
    assert hull[1][1] < 20


def test_mesh_lip_motion_stays_low_when_mouth_is_static_open() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    landmarks[13] = SimpleNamespace(x=0.50, y=0.50)
    landmarks[14] = SimpleNamespace(x=0.50, y=0.54)
    landmarks[61] = SimpleNamespace(x=0.40, y=0.52)
    landmarks[291] = SimpleNamespace(x=0.60, y=0.52)
    assert adapter._estimate_mesh_lip_motion(landmarks, width=200, height=200) == 0.0
    assert adapter._estimate_mesh_lip_motion(landmarks, width=200, height=200) == 0.0


def test_mesh_lip_motion_increases_when_mouth_moves() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    landmarks[13] = SimpleNamespace(x=0.50, y=0.50)
    landmarks[14] = SimpleNamespace(x=0.50, y=0.52)
    landmarks[61] = SimpleNamespace(x=0.40, y=0.52)
    landmarks[291] = SimpleNamespace(x=0.60, y=0.52)
    adapter._estimate_mesh_lip_motion(landmarks, width=200, height=200)
    landmarks[14] = SimpleNamespace(x=0.50, y=0.56)
    assert adapter._estimate_mesh_lip_motion(landmarks, width=200, height=200) > 0.1


def test_mouth_is_occluded_when_hand_landmarks_overlap_lip_region() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_lip_boxes(landmarks)
    rgb = np.full((200, 200, 3), 180, dtype=np.uint8)
    hand = [SimpleNamespace(x=0.50, y=0.64)]
    hand_result = SimpleNamespace(hand_landmarks=[hand])
    assert adapter._mouth_is_occluded(landmarks, rgb, hand_result, width=200, height=200) is True


def test_mouth_visual_evidence_is_low_for_generic_occluder() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    rgb = np.full((120, 160, 3), 125, dtype=np.uint8)
    mouth_points = [(58, 70), (102, 70), (102, 88), (58, 88)]
    assert adapter._mouth_visual_evidence_score(rgb, mouth_points) < MOUTH_OCCLUSION_EVIDENCE_THRESHOLD


def test_mouth_visual_evidence_is_high_when_lip_edges_are_visible() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    rgb = np.full((120, 160, 3), 185, dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.ellipse(rgb, (80, 78), (25, 8), 0, 0, 360, (65, 65, 65), 2)
    cv2.line(rgb, (58, 78), (102, 78), (45, 45, 45), 1)
    mouth_points = [(58, 70), (102, 70), (102, 88), (58, 88)]
    assert adapter._mouth_visual_evidence_score(rgb, mouth_points) >= MOUTH_OCCLUSION_EVIDENCE_THRESHOLD


def test_multi_person_candidate_selection_prefers_clear_lip_motion() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    candidates = [
        {"candidate_id": "local_user_0", "candidate_score": 0.30, "lip_motion": 0.01},
        {"candidate_id": "local_user_1", "candidate_score": 0.58, "lip_motion": 0.05},
    ]
    selected, ambiguous = adapter._select_face_mesh_candidate(candidates)
    assert selected is candidates[1]
    assert ambiguous is False


def test_multi_person_candidate_selection_rejects_close_scores() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    candidates = [
        {"candidate_id": "local_user_0", "candidate_score": 0.52, "lip_motion": 0.04},
        {"candidate_id": "local_user_1", "candidate_score": 0.47, "lip_motion": 0.035},
    ]
    selected, ambiguous = adapter._select_face_mesh_candidate(candidates)
    assert selected is None
    assert ambiguous is True


def test_multi_person_candidate_selection_allows_lip_dominance_with_close_scores() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_mouth_motion=0.008))
    candidates = [
        {"candidate_id": "local_user_0", "candidate_score": 0.52, "lip_motion": 0.018},
        {"candidate_id": "local_user_1", "candidate_score": 0.48, "lip_motion": 0.004},
    ]
    selected, ambiguous = adapter._select_face_mesh_candidate(candidates)
    assert selected is candidates[0]
    assert ambiguous is False


def test_candidate_score_requires_lip_motion() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_mouth_motion=0.008))
    assert (
        adapter._candidate_score(
            lip_motion=0.0,
            gaze_score=1.0,
            head_yaw_deg=0.0,
            direction_deg=0.0,
            distance_m=0.8,
            mouth_occluded=False,
        )
        == 0.0
    )


def test_mesh_gaze_score_is_high_when_irises_are_centered() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    for index in range(468, 473):
        landmarks[index] = SimpleNamespace(x=0.417, y=0.47)
    for index in range(473, 478):
        landmarks[index] = SimpleNamespace(x=0.597, y=0.47)
    assert adapter._estimate_mesh_gaze_score(landmarks, width=200, height=200, face_center_score=1.0) > 0.7


def test_mesh_gaze_score_is_low_when_irises_are_off_center() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    for index in range(468, 473):
        landmarks[index] = SimpleNamespace(x=0.48, y=0.47)
    for index in range(473, 478):
        landmarks[index] = SimpleNamespace(x=0.66, y=0.47)
    assert adapter._estimate_mesh_gaze_score(landmarks, width=200, height=200, face_center_score=1.0) < 0.55


def test_mesh_gaze_score_is_low_when_eyes_are_closed_even_if_irises_centered() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_closed_eye_boxes(landmarks)
    for index in range(468, 473):
        landmarks[index] = SimpleNamespace(x=0.417, y=0.471)
    for index in range(473, 478):
        landmarks[index] = SimpleNamespace(x=0.597, y=0.471)
    assert adapter._estimate_mesh_gaze_score(landmarks, width=200, height=200, face_center_score=1.0) == 0.0


def test_eye_openness_score_detects_closed_eye() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    open_eye = [(0, 0), (40, 0), (40, 10), (0, 10)]
    closed_eye = [(0, 4), (40, 4), (40, 6), (0, 6)]
    assert adapter._eye_openness_score(open_eye) > 0.45
    assert adapter._eye_openness_score(closed_eye) < 0.45


def test_eyes_are_occluded_when_hand_landmarks_overlap_eye_region() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    hand = [SimpleNamespace(x=0.42, y=0.47)]
    hand_result = SimpleNamespace(hand_landmarks=[hand])
    assert adapter._eyes_are_occluded_by_hand(landmarks, hand_result, width=200, height=200) is True


def test_eyes_are_not_occluded_when_hand_landmarks_are_away_from_eye_region() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    hand = [SimpleNamespace(x=0.90, y=0.90)]
    hand_result = SimpleNamespace(hand_landmarks=[hand])
    assert adapter._eyes_are_occluded_by_hand(landmarks, hand_result, width=200, height=200) is False


def test_eye_occlusion_state_handles_one_eye_only() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    rgb = np.full((200, 200, 3), 210, dtype=np.uint8)
    rgb[90:98, 82:90] = 30
    rgb[90:98, 118:126] = 30
    hand = [SimpleNamespace(x=0.42, y=0.47)]
    hand_result = SimpleNamespace(hand_landmarks=[hand])
    occlusion = adapter._eye_occlusion_state(landmarks, rgb, hand_result, width=200, height=200)
    assert occlusion == {"left": True, "right": False}


def test_eye_visual_evidence_is_low_for_generic_occluder() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    rgb = np.full((80, 120, 3), 120, dtype=np.uint8)
    eye_points = [(30, 30), (70, 30), (70, 42), (30, 42)]
    iris_points = [(48, 34), (52, 34), (52, 38), (48, 38)]
    assert adapter._eye_visual_evidence_score(rgb, eye_points, iris_points) < EYE_OCCLUSION_EVIDENCE_THRESHOLD


def test_eye_visual_evidence_is_high_when_iris_contrast_is_visible() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    rgb = np.full((80, 120, 3), 210, dtype=np.uint8)
    rgb[32:41, 46:55] = 30
    eye_points = [(30, 30), (70, 30), (70, 42), (30, 42)]
    iris_points = [(48, 34), (52, 34), (52, 38), (48, 38)]
    assert adapter._eye_visual_evidence_score(rgb, eye_points, iris_points) >= EYE_OCCLUSION_EVIDENCE_THRESHOLD


def test_eye_with_glasses_like_low_contrast_is_not_treated_as_occluded() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    landmarks = _landmarks()
    _set_eye_boxes(landmarks)
    for index in range(468, 473):
        landmarks[index] = SimpleNamespace(x=0.417, y=0.47)
    rgb = np.full((200, 200, 3), 170, dtype=np.uint8)
    rgb[89:100, 80:91] = 135
    rgb[87:89, 78:94] = 230
    assert adapter._eye_is_occluded(
        landmarks,
        rgb,
        hand_result=None,
        width=200,
        height=200,
        eye_indices=LEFT_EYE_POINTS,
        iris_indices=range(468, 473),
    ) is False


def test_lip_and_gaze_overlay_conditions_are_independent() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig(min_mouth_motion=0.02))
    assert adapter._lip_is_moving(0.01) is False
    assert adapter._lip_is_moving(0.03) is True
    assert adapter._eyes_are_gazing(0.54) is False
    assert adapter._eyes_are_gazing(0.55) is True


def test_stable_gaze_state_uses_hysteresis_near_threshold() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    candidate_id = "local_user_0"
    assert adapter._stable_gaze_state(candidate_id, 0.56) is True
    assert adapter._stable_gaze_state(candidate_id, 0.53) is True
    assert adapter._stable_gaze_state(candidate_id, 0.49) is True
    assert adapter._stable_gaze_state(candidate_id, 0.47) is False


def test_stabilized_gaze_score_smooths_small_jitter() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    candidate_id = "local_user_0"
    scores = [
        adapter._stabilized_gaze_score(candidate_id, score, {"left": False, "right": False})
        for score in (0.58, 0.52, 0.57, 0.53)
    ]
    assert min(scores[1:]) > 0.54


def test_stabilized_gaze_score_clears_when_both_eyes_occluded() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    candidate_id = "local_user_0"
    adapter._stabilized_gaze_score(candidate_id, 0.70, {"left": False, "right": False})
    adapter._stable_gaze_state(candidate_id, 0.70)
    assert adapter._stabilized_gaze_score(candidate_id, 0.70, {"left": True, "right": True}) == 0.0
    assert adapter._gaze_states.get(candidate_id) is None


def test_head_angle_condition_turns_invalid_over_30_degrees() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    assert adapter._head_angle_is_valid(30.0) is True
    assert adapter._head_angle_is_valid(30.1) is False
    assert adapter._head_angle_is_valid(-30.1) is False


def test_face_matrix_yaw_uses_3d_rotation_matrix() -> None:
    adapter = LocalCameraMicAdapter(LocalInputConfig())
    yaw_rad = np.deg2rad(35.0)
    matrix = np.array(
        [
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    result = SimpleNamespace(facial_transformation_matrixes=[matrix])
    assert adapter._face_matrix_yaw_deg(result) == pytest.approx(35.0)
