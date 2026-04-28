# Loona 无唤醒词唤醒监控工具

这是第一版 Python + Qt 工程，用于展示无唤醒词语音唤醒判断结果。

## 已实现

- 黑色主题 Qt 主界面。
- 本地摄像头实时预览面板。
- IDLE / WAKEUP 状态显示框实时刷新。
- MediaPipe FaceMesh 人脸、嘴唇、眼睛关键点检测。
- 多人同框稳定候选人跟踪，避免把 A 的注视和 B 的声音拼成误唤醒。
- 本地麦克风人声能量、周期性和语音质量检测，降低吹气、气流声、短促咳嗽等非语音误触发。
- 句内意图一致性、目标稳定性、声源方向和人脸方向匹配判断。
- Mock 多模态数据流。
- Live UDP 多模态数据接入。
- 规则加权版 WakeupDecisionEngine。

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
set PYTHONPATH=src
python -m loona_wakeup.main
```

在 Git Bash 中可使用：

```bash
PYTHONPATH=src python -m loona_wakeup.main
```

## 后续接入

真实底层能力接入时，替换 `src/loona_wakeup/adapters/mock_adapter.py`，按 `MultimodalFrame` 输出即可。

当前摄像头面板用于实时预览系统默认摄像头。默认使用 `local` 模式，优先通过 MediaPipe FaceMesh 检测人脸、嘴唇和眼睛关键点，OpenCV Haar 作为兜底；同时从本机麦克风检测人声能量，用这些本地结果驱动唤醒判断。

FaceMesh 会绘制更贴合真实轮廓的人脸、嘴唇和眼睛线条；满足当前识别条件时显示绿色，不满足时显示红色。

多人同时出现在画面中时，本地模式最多检测 3 张脸，并通过人脸位置给每个人分配稳定 `track_id`，独立计算朝向、注视、嘴唇运动、遮挡和距离。系统只选择一个主说话候选人进入唤醒判断；非主候选人只显示低亮轮廓。如果候选人分数太接近，或一句话期间主候选人发生切换，会判定为 `multi_person_ambiguous` 并拒绝唤醒。

多人场景不会跨人合并信号：A 在注视 Loona、B 在旁边说话时，不会把 A 的视觉意图和 B 的声音拼成一次唤醒。

MediaPipe 模型文件位于 [assets/models/face_landmarker.task](assets/models/face_landmarker.task)。如果该文件缺失，程序会自动回退到 OpenCV Haar 检测，但识别率会明显下降。

`live` 模式仍可用于接收外部底层能力通过 UDP 发来的头部朝向、视线、唇动、声源、距离等真实输出。

`mock` 模式只用于演示界面和判断链路，会周期性模拟“有人对 Loona 说话”的场景；不要用 `mock` 模式判断现实无人说话时的误唤醒率。

## 接入真实底层多模态输出

确认 [configs/default.yaml](configs/default.yaml) 中的运行模式为：

```yaml
runtime:
	mode: local
```

本地模式配置：

```yaml
local_input:
	camera_index: 0
	min_audio_energy: 0.015
	min_mouth_motion: 0.008
```

如果摄像头画面正常但说话不唤醒，可以先适当降低 `min_audio_energy`；如果没说话也容易唤醒，可以提高 `min_audio_energy` 或 `min_mouth_motion`。

## 接入外部底层多模态输出

如果不用本地检测，而是使用外部底层能力，将运行模式改为：

```yaml
runtime:
	mode: live
```

默认监听 UDP：`127.0.0.1:8765`。

底层能力模块向该端口发送 JSON，例如：

```json
{
	"timestamp_ms": 1777286400000,
	"user_id": "user_01",
	"has_voice": true,
	"voice_energy": 0.88,
	"speech_like_score": 0.91,
	"sound_direction_deg": 0,
	"sound_distance_m": 0.85,
	"face_visible": true,
	"head_yaw_deg": 0,
	"head_pitch_deg": 2,
	"gaze_to_loona_score": 0.82,
	"lip_movement_score": 0.78,
	"is_attention_target": true,
	"scene_type": "live",
	"background_audio_score": 0.05
}
```

也支持常见别名字段：`target_user_id`、`voice`、`vad`、`direction_deg`、`visual_direction_deg`、`sound_face_match`、`distance_m`、`head_yaw`、`gaze_score`、`lip_score`、`attention_target`、`background_score`。

默认策略偏保守：需要可靠人声、可见人脸和唇动同步，才允许触发唤醒。若底层暂时无法稳定输出 `face_visible` 或 `lip_movement_score`，可以在 [configs/default.yaml](configs/default.yaml) 中临时关闭：

```yaml
wakeup:
	require_face_visible: false
	require_lip_sync: false
```

不建议在线上关闭这两个门槛，否则没人说话、背景音或电脑外放更容易造成误唤醒。

唤醒还需要连续多帧确认，默认配置为：

```yaml
wakeup:
	min_consecutive_wakeup_frames: 3
	min_wakeup_voice_ms: 320
	min_wakeup_voice_frames: 3
	min_intent_consistency_score: 0.45
	min_target_stability_score: 0.75
	min_sound_face_match_score: 0.45
```

这可以过滤底层人声、唇动或视线的单帧抖动，也会拒绝咳嗽这类过短的非语音爆发声、句内意图不连续、多人目标不稳定以及声源方向和人脸方向明显不一致的情况。

可用测试脚本模拟底层输出：

```bash
PYTHONPATH=src e:/1_keyi/loona_wakeup/.venv/Scripts/python.exe tools/send_live_frame.py --mode alternate
```
