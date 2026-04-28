# Loona 无唤醒词语音唤醒 Python + Qt 实现方案

## 1. 当前目标

本工程用于实现 Loona 的无唤醒词语音唤醒原型：在持续监听状态下，结合人声、人脸、朝向、注视、唇动、声源方向和距离等信息，判断用户是否在对 Loona 说话，并在一句话说完后输出唤醒结果。

当前版本聚焦一个完整闭环：

1. 本机摄像头和麦克风实时输入。
2. 摄像头画面中显示人脸、嘴唇、注视状态轮廓。
3. 等用户一句话说完后，再判断是否需要唤醒。
4. UI 只保留 Camera 和 IDLE / WAKEUP 状态框。
5. 支持外部底层能力通过 UDP 发送多模态结果。
6. 保留 Mock 模式用于演示和测试链路。

## 2. 技术选型

### 2.1 主语言

使用 Python。

原因：

- 便于快速验证无唤醒词多模态融合策略。
- 便于接入本地摄像头、麦克风和外部 UDP 数据。
- 便于用 pytest 做规则测试和回归验证。
- 后续可将稳定后的判断逻辑迁移到设备侧服务或 C++ 实现。

### 2.2 GUI 框架

使用 PySide6。

当前 UI 采用 Qt Widgets，黑色主题，主界面只显示运行状态、IDLE / WAKEUP 状态和 Camera 画面。

### 2.3 当前依赖

```text
PySide6>=6.6
PyYAML>=6.0
pytest>=8.0
numpy>=1.26
opencv-python>=4.9
sounddevice>=0.4
mediapipe>=0.10.14
```

说明：

- `mediapipe` 用于 FaceMesh 人脸、嘴唇和眼睛关键点检测。
- `opencv-python` 用于本地摄像头采集、兜底 Haar 检测和画面绘制。
- `sounddevice` 用于本机麦克风能量检测。
- `numpy` 用于音频 RMS 和图像差分计算。
- `PyYAML` 用于加载 `configs/default.yaml`。

## 3. 当前总体架构

```text
输入层
  ├─ LocalCameraMicAdapter：本机摄像头 + 麦克风
  ├─ LiveUdpAdapter：外部底层能力 UDP JSON
  └─ MockAdapter：演示数据流
        ↓
MultimodalFrame 统一数据帧
        ↓
UtteranceGate 话语段门控
  ├─ 有声音时缓存帧
  ├─ 静音达到阈值后认为一句话结束
  └─ 输出整句话的帧序列
        ↓
WakeupDecisionEngine
  ├─ 汇总整句话多模态特征
  ├─ 硬性拒绝条件
  ├─ 加权评分
  └─ 输出 WakeupDecision
        ↓
Qt MainWindow
  ├─ Camera 画面
  └─ IDLE / WAKEUP 状态
```

当前应用入口在 `src/loona_wakeup/app.py`：

1. 创建 `QApplication`。
2. 加载配置。
3. 创建 `MainWindow`。
4. 创建 `WakeupDecisionEngine`。
5. 创建 `UtteranceGate`。
6. 根据 `runtime.mode` 创建 Adapter。
7. Adapter 推送 `MultimodalFrame`。
8. `UtteranceGate` 等一句话结束。
9. `WakeupDecisionEngine.decide_utterance()` 输出判断。
10. UI 更新 IDLE / WAKEUP。

## 4. 输入模式

### 4.1 local 模式

默认模式为 `local`。

`LocalCameraMicAdapter` 直接使用本机摄像头和麦克风：

- OpenCV `VideoCapture` 读取摄像头。
- MediaPipe FaceMesh 优先检测 468 点人脸关键点。
- FaceMesh 使用 `assets/models/face_landmarker.task` 模型文件。
- 基于 FaceMesh 关键点绘制真实人脸、眼睛和嘴唇轮廓。
- 基于嘴唇关键点开合和变化估计唇动。
- 基于人脸中心和眼部关键点估计注视 Loona 的程度。
- FaceMesh 失败时，回退到 OpenCV Haar 人脸/眼睛检测。
- Haar 回退路径中，未检测到眼睛时使用正脸程度作为注视估计兜底。
- `sounddevice.InputStream` 计算麦克风 RMS 能量。
- 输出 `MultimodalFrame` 和摄像头 `QImage` 预览帧。

画面覆盖层规则：

- 人脸轮廓：FaceMesh 面部外轮廓关键点连线。
- 嘴唇轮廓：FaceMesh 内外唇关键点连线。
- 注视轮廓：FaceMesh 左右眼关键点连线。
- 满足条件显示绿色。
- 不满足条件显示红色。
- 没有人脸时，画面中心显示红色人脸轮廓提示。

### 4.2 live 模式

`LiveUdpAdapter` 监听 UDP JSON，默认地址：

```text
127.0.0.1:8765
```

外部底层能力可以发送统一字段：

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

也支持常见别名字段，例如 `target_user_id`、`voice`、`vad`、`direction_deg`、`distance_m`、`head_yaw`、`gaze_score`、`lip_score`、`attention_target`、`background_score`。

### 4.3 mock 模式

`MockAdapter` 用于演示 UI 和判断链路。该模式会周期性模拟有效唤醒场景，不适合用于评估真实无人场景的误唤醒率。

## 5. 数据结构

当前数据结构使用 `dataclass(slots=True)`，定义在 `src/loona_wakeup/models.py`。

### 5.1 RunMode

```python
class RunMode(str, Enum):
    MOCK = "mock"
    LOCAL = "local"
    REPLAY = "replay"
    LIVE = "live"
```

当前实际支持 `mock`、`local`、`live`；`replay` 仅作为预留枚举。

### 5.2 MultimodalFrame

```python
@dataclass(slots=True)
class MultimodalFrame:
    timestamp_ms: int
    user_id: str | None = None
    has_voice: bool = False
    voice_energy: float = 0.0
    speech_like_score: float = 0.0
    sound_direction_deg: float | None = None
    sound_distance_m: float | None = None
    face_visible: bool = False
    head_yaw_deg: float | None = None
    head_pitch_deg: float | None = None
    gaze_to_loona_score: float = 0.0
    lip_movement_score: float = 0.0
    is_attention_target: bool = False
    scene_type: str = "unknown"
    background_audio_score: float = 0.0
```

### 5.3 WakeupDecision

```python
@dataclass(slots=True)
class WakeupDecision:
    timestamp_ms: int
    wakeup: bool
    confidence: float
    target_user_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
```

## 6. 一句话结束后再判断

当前工程不再边说边触发唤醒，而是通过 `UtteranceGate` 等一句话结束后再判断。

话语段逻辑：

1. 当前帧有人声时，缓存 `MultimodalFrame`。
2. 用户仍在说话时，不调用唤醒判断。
3. 检测到静音持续达到 `utterance_end_silence_ms` 后，认为一句话结束。
4. 如果话语时长短于 `min_utterance_ms`，丢弃该片段。
5. 如果话语超过 `max_utterance_ms`，强制结算一次。
6. 输出整句话帧序列给 `WakeupDecisionEngine.decide_utterance()`。

默认配置中：

```yaml
wakeup:
  utterance_end_silence_ms: 700
  min_utterance_ms: 300
  max_utterance_ms: 8000
```

说明：

- `utterance_end_silence_ms` 越小，响应越快，但句中停顿更容易被当成结束。
- `utterance_end_silence_ms` 越大，判断更稳，但唤醒反馈会更慢。

## 7. 唤醒融合判断

`WakeupDecisionEngine` 支持两种判断入口：

- `decide(frame)`：单帧判断，保留给测试或兼容逻辑。
- `decide_utterance(frames)`：当前应用实际使用的整句话判断。

整句话判断会先聚合多帧特征：

- 取最高语音能量和最高 speech-like 分数。
- 取最强 gaze / lip 视觉证据。
- 取最优距离。
- 只要话语中出现过可见人脸、注意力目标，即作为整句话证据。
- 输出聚合后的 `MultimodalFrame` 再进入硬拒绝和加权评分。

### 7.1 硬性拒绝

当前硬拒绝条件包括：

- `no_reliable_voice`：没有可靠人声。
- `face_not_visible`：配置要求人脸可见但当前不可见。
- `distance_too_far`：距离超过最大交互距离。
- `background_audio_without_lip_sync`：背景音明显且无唇动同步。
- `no_visible_target`：没有可见目标且声源方向不明确。
- `no_intent_signal`：朝向、注视、唇动、注意力均不足。
- `no_lip_voice_sync`：配置要求唇声同步但唇动不足。

### 7.2 加权评分

默认权重：

```yaml
weights:
  voice_score: 0.25
  sound_position: 0.20
  distance_score: 0.15
  head_pose_score: 0.15
  gaze_score: 0.15
  lip_score: 0.20
  attention_bonus: 0.10
  background_penalty: -0.20
```

满足硬条件后，若归一化置信度达到 `min_confidence`，输出 `wakeup=true`。

默认阈值：

```yaml
wakeup:
  min_confidence: 0.72
  min_voice_score: 0.45
  max_distance_m: 2.0
  min_gaze_score: 0.45
  min_lip_score: 0.35
  min_visual_intent_score: 0.55
  require_face_visible: true
  require_lip_sync: true
```

单帧 `decide(frame)` 仍保留连续多帧确认：

```yaml
wakeup:
  min_consecutive_wakeup_frames: 3
```

当前应用使用整句话判断，句末只判断一次，因此 `decide_utterance()` 不再要求连续三次确认。

## 8. UI 设计

当前主窗口为极简黑色主题。

界面结构：

```text
┌────────────────────────────────────────────┐
│ Loona Wakeup Monitor       LOCAL ● RUNNING │
├────────────────────────────────────────────┤
│ IDLE / WAKEUP                              │
├────────────────────────────────────────────┤
│ Camera                                     │
│   摄像头画面 + 人脸/嘴唇/注视轮廓          │
└────────────────────────────────────────────┘
```

当前 UI 行为：

- `CameraPreview` 不再自己打开摄像头，只显示 Adapter 传来的 `QImage`。
- `LocalCameraMicAdapter.preview_ready` 连接到 `MainWindow.update_camera_frame()`。
- `frame_ready` 只驱动话语段和判断，不直接改变画面。
- `WAKEUP` 显示由 `QTimer` 保持一段时间；当前代码中保持时间为 1000ms。
- Pause / Resume 按钮可以暂停判断链路。

颜色规范：

```text
背景色：#0B0D10
面板色：#12161C
边框色：#242A33
主文字：#E6EDF3
次文字：#8B949E
唤醒成功：#3FB950
拒绝/风险：#F85149
```

## 9. 配置文件

默认配置文件为 `configs/default.yaml`。

当前默认运行模式：

```yaml
runtime:
  mode: local
  ui_refresh_ms: 100
  decision_interval_ms: 100
```

本地输入配置：

```yaml
local_input:
  camera_index: 0
  frame_interval_ms: 80
  audio_sample_rate: 16000
  audio_block_size: 1024
  min_audio_energy: 0.015
  min_mouth_motion: 0.008
```

Live UDP 配置：

```yaml
live_udp:
  host: 127.0.0.1
  port: 8765
  poll_interval_ms: 30
  max_datagram_size: 65535
```

## 10. 当前工程目录

```text
loona_wakeup/
  README.md
  Python_Qt实现方案.md
  requirements.txt
  pyproject.toml
  configs/
    default.yaml
  assets/
    models/
      face_landmarker.task
  src/
    loona_wakeup/
      __init__.py
      app.py
      config.py
      main.py
      models.py
      adapters/
        __init__.py
        live_udp_adapter.py
        local_camera_mic_adapter.py
        mock_adapter.py
      engine/
        __init__.py
        decision_engine.py
        utterance_gate.py
      ui/
        __init__.py
        camera_preview.py
        main_window.py
        theme.py
        widgets.py
  tests/
    test_decision_engine.py
    test_live_udp_adapter.py
    test_local_camera_mic_adapter.py
    test_utterance_gate.py
  tools/
    send_live_frame.py
```

## 11. 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

Windows CMD：

```bat
set PYTHONPATH=src
python -m loona_wakeup.main
```

Git Bash：

```bash
PYTHONPATH=src python -m loona_wakeup.main
```

当前开发环境中可直接使用：

```bash
PYTHONPATH=src e:/1_keyi/loona_wakeup/.venv/Scripts/python.exe -m loona_wakeup.main
```

## 12. 测试与验证

当前测试覆盖：

- 多模态融合判断。
- 背景音、人脸缺失、唇声不同步等拒绝条件。
- 整句话聚合判断。
- UDP JSON 字段映射。
- 本地麦克风 speech score。
- 戴眼镜时的注视兜底逻辑。
- 眼睛区域过滤。
- 话语段门控：说话过程中不输出，静音结束后输出。

验证命令：

```bash
PYTHONPATH=src e:/1_keyi/loona_wakeup/.venv/Scripts/python.exe -m compileall -q src tests tools
PYTHONPATH=src e:/1_keyi/loona_wakeup/.venv/Scripts/python.exe -m pytest tests -q
```

## 13. 当前已实现内容

- Python + PySide6 黑色主题桌面工具。
- 极简 UI：Camera + IDLE / WAKEUP。
- 本地摄像头预览。
- 本地麦克风人声能量检测。
- MediaPipe FaceMesh 高精度人脸、嘴唇和眼睛关键点检测。
- OpenCV Haar 兜底检测。
- 戴眼镜场景注视兜底。
- 基于嘴唇关键点的嘴部运动估计。
- 人脸、嘴唇、注视红绿关键点轮廓覆盖层。
- 等一句话说完后再判断唤醒。
- 多模态硬拒绝和加权评分。
- 外部 UDP 多模态数据接入。
- Mock 演示数据。
- pytest 回归测试。

## 14. 暂不实现内容

以下内容仍不属于当前版本范围：

- 固定唤醒词识别。
- 唤醒后的 ASR / NLU / TTS 链路。
- Loona 播报过程中的打断和恢复播报。
- 完整多人身份识别。
- 基于深度学习的精确 FaceMesh / 唇形关键点模型。
- 完整数据标注和训练平台。

## 15. 后续可优化方向

1. 使用更多真实摄像头样本调校 FaceMesh 注视和唇动阈值。
2. 增加真实日志记录与回放，沉淀误唤醒和唤不醒样本。
3. 为 `utterance_end_silence_ms`、`min_audio_energy`、`min_mouth_motion` 增加 UI 调参入口。
4. 增加唤醒事件输出接口，例如本地 socket、HTTP 或进程消息。
5. 根据实际设备麦克风和摄像头视角，重新标定距离和声源方向估计。
