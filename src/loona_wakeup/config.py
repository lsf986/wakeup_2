from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from loona_wakeup.models import AppConfig, LiveUdpConfig, LocalInputConfig, LoggingConfig, RunMode, RuntimeConfig, WakeupConfig, WeightConfig

T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    if not is_dataclass(instance):
        return instance
    known_fields = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key in known_fields:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config = AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if runtime_raw := raw.get("runtime"):
        if "mode" in runtime_raw:
            runtime_raw = dict(runtime_raw)
            runtime_raw["mode"] = RunMode(runtime_raw["mode"])
        config.runtime = _update_dataclass(RuntimeConfig(), runtime_raw)
    if wakeup_raw := raw.get("wakeup"):
        config.wakeup = _update_dataclass(WakeupConfig(), wakeup_raw)
    if weights_raw := raw.get("weights"):
        config.weights = _update_dataclass(WeightConfig(), weights_raw)
    if logging_raw := raw.get("logging"):
        config.logging = _update_dataclass(LoggingConfig(), logging_raw)
    if live_udp_raw := raw.get("live_udp"):
        config.live_udp = _update_dataclass(LiveUdpConfig(), live_udp_raw)
    if local_input_raw := raw.get("local_input"):
        config.local_input = _update_dataclass(LocalInputConfig(), local_input_raw)

    return config
