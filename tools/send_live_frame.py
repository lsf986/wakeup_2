from __future__ import annotations

import argparse
import json
import socket
import time


WAKEUP_FRAME = {
    "user_id": "user_01",
    "has_voice": True,
    "voice_energy": 0.88,
    "speech_like_score": 0.91,
    "sound_direction_deg": 0,
    "sound_distance_m": 0.85,
    "face_visible": True,
    "head_yaw_deg": 0,
    "head_pitch_deg": 2,
    "gaze_to_loona_score": 0.82,
    "lip_movement_score": 0.78,
    "is_attention_target": True,
    "scene_type": "live_sample_wakeup",
    "background_audio_score": 0.05,
}

REJECT_FRAME = {
    "user_id": "user_02",
    "has_voice": True,
    "voice_energy": 0.72,
    "speech_like_score": 0.70,
    "sound_direction_deg": 68,
    "sound_distance_m": 3.1,
    "face_visible": True,
    "head_yaw_deg": 35,
    "head_pitch_deg": 0,
    "gaze_to_loona_score": 0.08,
    "lip_movement_score": 0.02,
    "is_attention_target": False,
    "scene_type": "live_sample_background",
    "background_audio_score": 0.86,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["wakeup", "reject", "alternate"], default="alternate")
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    index = 0
    while True:
        if args.mode == "wakeup":
            payload = dict(WAKEUP_FRAME)
        elif args.mode == "reject":
            payload = dict(REJECT_FRAME)
        else:
            payload = dict(WAKEUP_FRAME if (index // 20) % 2 == 0 else REJECT_FRAME)
        payload["timestamp_ms"] = int(time.time() * 1000)
        socket_client.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
        index += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
