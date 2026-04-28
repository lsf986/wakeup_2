from loona_wakeup.adapters.live_udp_adapter import frame_from_payload


def test_frame_from_payload_supports_alias_fields() -> None:
    frame = frame_from_payload(
        {
            "ts": 1777286400000,
            "target_user_id": "user_01",
            "voice": True,
            "voice_energy": "0.8",
            "speech_like_score": 0.9,
            "direction_deg": -5,
            "distance_m": 0.8,
            "face_visible": True,
            "head_yaw": 3,
            "gaze_score": 0.7,
            "lip_score": 0.6,
            "attention_target": True,
            "track_id": "local_user_0",
            "multi_person_count": "2",
            "multi_person_ambiguous": "true",
            "background_score": 0.1,
        }
    )

    assert frame.user_id == "user_01"
    assert frame.has_voice is True
    assert frame.sound_direction_deg == -5
    assert frame.sound_distance_m == 0.8
    assert frame.gaze_to_loona_score == 0.7
    assert frame.lip_movement_score == 0.6
    assert frame.is_attention_target is True
    assert frame.target_track_id == "local_user_0"
    assert frame.multi_person_count == 2
    assert frame.multi_person_ambiguous is True
