# Loona 无唤醒词语音唤醒 Python + Qt 实现方案

## 1. 方案目标

基于现有底层能力，实现一个 Python + Qt 桌面原型，用于持续接收音频与多模态信号，判断用户是否正在对 Loona 说话，并在判断成立时输出唤醒事件。

本方案重点解决：

1. 接入底层多模态信号。
2. 对朝向、视线、唇动、声源、距离等信号进行融合判断。
3. 输出 `wakeup=true/false`、置信度、触发依据、目标用户和拒绝原因。
4. 用黑色极简界面展示关键状态和调试信息。
5. 支持后续用真实场景数据调参、回放和验证。

## 2. 技术选型

### 2.1 主语言

使用 Python。

原因：

- 便于快速接入现有底层能力。
- 便于调试多模态融合规则和阈值。
- 便于做日志回放、数据统计和测试集验证。
- 后续可将稳定后的判断逻辑迁移到 C++ / ONNX / 设备侧服务。

### 2.2 GUI 框架

推荐使用 PySide6。

原因：

- 官方 Qt for Python 绑定，许可和维护状态较清晰。
- 对 Qt Widgets、信号槽、多线程支持完整。
- 适合做简洁、稳定的桌面调试工具。
- 后续如果需要更复杂的界面，也可以扩展到 Qt Quick。

### 2.3 核心依赖

建议依赖：

```text
PySide6
pydantic
numpy
pyyaml
loguru
```

可选依赖：

```text
opencv-python
sounddevice
websockets
fastapi
uvicorn
```

说明：

- 如果底层能力已经通过 SDK / 本地进程 / Socket 输出，则不需要在本工程里实现视觉或音频算法。
- 如果需要本地模拟输入，可增加 `opencv-python` 和 `sounddevice`。
- 如果底层能力通过 HTTP / WebSocket 输出，可增加 `fastapi` / `websockets`。

## 3. 总体架构

```text
底层能力层
  ├─ 音频 / VAD / 声源方向 / 距离
  ├─ 人脸 / 头部朝向 / 视线
  ├─ 唇动检测
  └─ 当前注意力目标
        ↓
信号接入层 Signal Adapter
        ↓
多模态状态缓存 State Store
        ↓
唤醒融合判断器 Wakeup Decision Engine
        ↓
唤醒事件 Wakeup Event
        ↓
Qt 黑色主题调试界面 + 日志记录
```

本工程不重新实现底层算法，只把底层输出转为统一格式，然后进行融合判断。

## 4. 模块划分

### 4.1 Signal Adapter

负责接入底层能力输出，并统一转换为内部数据结构。

支持三种输入模式：

1. Mock 模式：使用模拟数据，便于没有设备时调试界面和判断逻辑。
2. File Replay 模式：读取录制好的 JSONL / CSV 日志，进行离线回放。
3. Live 模式：从底层 SDK、Socket、WebSocket 或本地进程读取实时信号。

输出统一为 `MultimodalFrame`。

### 4.2 State Store

负责维护最近一段时间的多模态状态。

核心职责：

- 按时间戳缓存音频、视觉、距离、注意力等信号。
- 对不同频率的信号做时间对齐。
- 为融合判断器提供最近窗口内的稳定特征。

建议窗口长度：`0.8s - 2.0s`。

### 4.3 Wakeup Decision Engine

负责判断用户是否正在对 Loona 说话。

第一版建议使用规则 + 权重评分，不直接训练模型。

判断思路：

1. 先做硬性过滤。
2. 再做多信号加权评分。
3. 根据阈值输出唤醒或拒绝。
4. 输出触发依据和拒绝原因，方便调试。

### 4.4 Event Bus

负责在模块之间分发状态和事件。

事件类型：

- `frame_received`：收到一帧多模态信号。
- `decision_updated`：唤醒判断结果更新。
- `wakeup_triggered`：触发唤醒事件。
- `wakeup_rejected`：拒绝唤醒。

### 4.5 Qt UI

负责显示当前关键状态。

界面只展示最关键的信息，不做复杂控制台。

## 5. 数据结构设计

### 5.1 MultimodalFrame

```python
class MultimodalFrame(BaseModel):
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
```

### 5.2 WakeupDecision

```python
class WakeupDecision(BaseModel):
    timestamp_ms: int
    wakeup: bool
    confidence: float
    target_user_id: str | None = None
    reasons: list[str] = []
    reject_reasons: list[str] = []
    raw_scores: dict[str, float] = {}
```

### 5.3 WakeupConfig

```python
class WakeupConfig(BaseModel):
    min_confidence: float = 0.72
    min_voice_score: float = 0.45
    max_distance_m: float = 2.0
    min_gaze_score: float = 0.45
    min_lip_score: float = 0.35
    min_attention_bonus: float = 0.15
    decision_window_ms: int = 1200
    cooldown_ms: int = 1500
```

## 6. 融合判断策略

### 6.1 硬性过滤

以下条件满足时，优先拒绝唤醒：

1. 没有人声：`has_voice=false`。
2. 声音距离超过有效交互距离。
3. 声音像背景音，且没有对应唇动。
4. 用户不可见，且没有其他强指向信号。
5. 只有短音，没有朝向、视线、唇动等辅助信号。
6. 多人场景中，声音来源不是当前注意力目标，且视觉指向不明确。

### 6.2 加权评分

建议第一版评分：

```text
voice_score        0.25
sound_position     0.20
distance_score     0.15
head_pose_score    0.15
gaze_score         0.15
lip_score          0.20
attention_bonus    0.10
background_penalty -0.20
```

最终分数：

```text
confidence = weighted_sum(features) - penalties
```

当 `confidence >= min_confidence` 时触发唤醒。

### 6.3 推荐触发条件

P0 场景下，以下组合应稳定触发：

1. 近距离人声 + 声源来自用户方向 + 面向 Loona + 唇动同步。
2. 近距离人声 + 当前注意力目标 + 明确看向 Loona。
3. 短句 + 面向 Loona + 唇动同步 + 距离有效。

### 6.4 推荐拒绝条件

以下组合应拒绝：

1. 只有人声，无朝向、无视线、无唇动。
2. 只有看向 Loona，但没有人声或唇动。
3. 远处人声或电脑外放音频。
4. 多人自然对话，目标用户不明确。
5. 用户对他人说话，中途短暂扫过 Loona。

## 7. 界面设计

### 7.1 设计原则

- 黑色主题。
- 信息密度适中，只显示关键参数。
- 不展示复杂图表，避免干扰判断。
- 主界面用于观察当前状态，不做繁重配置。
- 配置通过 YAML 文件管理，界面只提供少量开关。

### 7.2 主界面布局

建议窗口尺寸：`960 x 540`。

界面结构：

```text
┌────────────────────────────────────────────┐
│ Loona Wakeup Monitor             LIVE ●    │
├────────────────────────────────────────────┤
│                                            │
│  WAKEUP / IDLE                             │
│  Confidence: 0.83                          │
│  Target: user_01                           │
│                                            │
├──────────────────────┬─────────────────────┤
│ Signals              │ Decision             │
│ Voice: ON            │ Reasons              │
│ Direction: -8 deg    │ - voice_near          │
│ Distance: 0.8 m      │ - gaze_to_loona       │
│ Head: facing         │ - lip_synced          │
│ Gaze: 0.71           │ Reject                │
│ Lip: 0.66            │ -                     │
│ Attention: yes       │                       │
├──────────────────────┴─────────────────────┤
│ Recent events                              │
│ 12:00:01 wakeup true conf=0.83             │
│ 12:00:03 wakeup false reason=background    │
└────────────────────────────────────────────┘
```

### 7.3 颜色规范

黑色主题建议：

```text
背景色：#0B0D10
面板色：#12161C
边框色：#242A33
主文字：#E6EDF3
次文字：#8B949E
强调色：#2F81F7
唤醒成功：#3FB950
拒绝/风险：#F85149
警告：#D29922
```

### 7.4 界面显示字段

顶部状态：

- 当前模式：Mock / Replay / Live。
- 运行状态：Running / Paused。
- 当前唤醒状态：WAKEUP / IDLE。
- 当前置信度。

关键参数：

- Voice。
- Sound Direction。
- Distance。
- Head Pose。
- Gaze Score。
- Lip Score。
- Attention Target。
- Scene Type。

判断信息：

- Wakeup True / False。
- Confidence。
- Target User。
- Trigger Reasons。
- Reject Reasons。

最近事件：

- 最近 20 条唤醒判断。
- 每条显示时间、结果、置信度、主要原因。

## 8. 工程目录建议

```text
loona_wakeup/
  README.md
  requirements.txt
  configs/
    default.yaml
  src/
    loona_wakeup/
      __init__.py
      app.py
      main.py
      config.py
      models.py
      event_bus.py
      adapters/
        __init__.py
        base.py
        mock_adapter.py
        replay_adapter.py
        live_adapter.py
      engine/
        __init__.py
        state_store.py
        feature_extractor.py
        decision_engine.py
      ui/
        __init__.py
        main_window.py
        theme.py
        widgets.py
      logging/
        __init__.py
        decision_logger.py
  tests/
    test_decision_engine.py
    test_state_store.py
```

## 9. 主流程

```text
1. 启动 Qt 应用。
2. 加载 default.yaml 配置。
3. 初始化 Signal Adapter。
4. 初始化 State Store。
5. 初始化 Wakeup Decision Engine。
6. Adapter 持续推送 MultimodalFrame。
7. State Store 缓存并对齐最近窗口信号。
8. Decision Engine 计算 WakeupDecision。
9. UI 更新关键状态。
10. Logger 记录决策结果。
11. 当 wakeup=true 时输出唤醒事件。
```

## 10. 线程模型

Qt 主线程只负责界面渲染。

建议线程划分：

1. UI Thread：Qt 主线程。
2. Adapter Worker：读取底层信号。
3. Decision Worker：执行融合判断。
4. Logger Worker：异步写日志。

线程间通过 Qt Signal / Slot 或线程安全队列通信。

不要在 UI 线程里执行持续读取、网络 IO 或高频计算。

## 11. 配置文件示例

```yaml
runtime:
  mode: mock
  ui_refresh_ms: 100
  decision_interval_ms: 100

wakeup:
  min_confidence: 0.72
  min_voice_score: 0.45
  max_distance_m: 2.0
  min_gaze_score: 0.45
  min_lip_score: 0.35
  decision_window_ms: 1200
  cooldown_ms: 1500

weights:
  voice_score: 0.25
  sound_position: 0.20
  distance_score: 0.15
  head_pose_score: 0.15
  gaze_score: 0.15
  lip_score: 0.20
  attention_bonus: 0.10
  background_penalty: -0.20

logging:
  enabled: true
  path: logs/decisions.jsonl
```

## 12. 唤醒事件格式

建议输出 JSON：

```json
{
  "event": "wakeup_decision",
  "timestamp_ms": 1777286400000,
  "wakeup": true,
  "confidence": 0.83,
  "target_user_id": "user_01",
  "reasons": ["voice_near", "gaze_to_loona", "lip_synced"],
  "reject_reasons": [],
  "raw_scores": {
    "voice_score": 0.81,
    "distance_score": 0.92,
    "gaze_score": 0.71,
    "lip_score": 0.66
  }
}
```

拒绝示例：

```json
{
  "event": "wakeup_decision",
  "timestamp_ms": 1777286400100,
  "wakeup": false,
  "confidence": 0.31,
  "target_user_id": null,
  "reasons": ["voice_detected"],
  "reject_reasons": ["background_audio", "no_lip_sync", "not_attention_target"],
  "raw_scores": {
    "voice_score": 0.62,
    "distance_score": 0.20,
    "gaze_score": 0.05,
    "lip_score": 0.00
  }
}
```

## 13. 开发阶段规划

### 阶段 1：可运行框架

目标：搭出 Python + Qt 黑色主题界面，支持 Mock 数据流。

交付：

- 主窗口。
- Mock Adapter。
- 数据模型。
- 基础 Decision Engine。
- 决策结果实时刷新。

### 阶段 2：规则融合判断

目标：实现第一版多模态规则融合。

交付：

- 硬性过滤。
- 加权评分。
- 触发依据和拒绝原因。
- JSONL 决策日志。

### 阶段 3：接入真实底层能力

目标：替换 Mock Adapter，接入已有底层能力。

交付：

- Live Adapter。
- 时间戳对齐。
- 多人目标用户字段。
- 实时稳定性优化。

### 阶段 4：场景测试与调参

目标：基于 P0 / P1 / P2 case 验证误唤醒和唤醒成功率。

交付：

- Replay Adapter。
- 测试日志。
- 阈值调整。
- 误唤醒 / 唤不醒分析报告。

## 14. 第一版工程建议

第一版先做一个可运行的本地工具：

1. 使用 `PySide6` 做主界面。
2. 使用 `MockAdapter` 模拟底层输入。
3. 使用 `DecisionEngine` 实现规则评分。
4. 使用 `decision_logger` 写 JSONL 日志。
5. UI 只展示当前状态、关键参数、判断原因和最近事件。

这样可以先把判断链路、界面和调试方式跑通，再逐步替换成真实底层数据。

## 15. 暂不实现内容

以下内容不进入第一版工程：

- 固定唤醒词识别。
- 唤醒后的语音处理链路。
- Loona 输出过程中的再次开口处理。
- 复杂语义意图识别。
- 模型训练平台。
- 完整数据标注系统。

## 16. 下一步

如果确认该方案，可以进入工程生成阶段。建议先生成最小可运行版本：

1. 黑色主题 Qt 主界面。
2. Mock 数据实时变化。
3. 唤醒判断结果实时刷新。
4. 关键参数面板。
5. 最近事件列表。
6. 可配置阈值文件。
