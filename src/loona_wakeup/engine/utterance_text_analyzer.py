from __future__ import annotations

from dataclasses import dataclass
import re


DIRECT_ADDRESS_PATTERNS = (
    "loona",
    "luna",
    "露娜",
    "鲁娜",
    "你",
    "帮我",
    "帮一下",
    "过来",
    "来一下",
    "看这里",
    "听我说",
    "我们走",
    "能不能",
    "可以吗",
    "告诉我",
    "陪我",
    "打开",
    "关闭",
    "开始",
    "停一下",
    "怎么样",
)

SELF_TALK_PATTERNS = (
    "我想想",
    "我看看",
    "我在想",
    "怎么回事",
    "奇怪",
    "不对",
    "等一下",
    "好像",
    "应该是",
    "可能是",
    "算了",
    "原来是",
)

FILLER_PATTERNS = {"嗯", "啊", "呃", "哦", "哎", "唉", "额", "呃呃", "嗯嗯"}
QUESTION_MARKERS = ("?", "？", "吗", "呢", "是不是", "能不能", "可不可以", "怎么样")
COMMAND_MARKERS = ("帮", "打开", "关闭", "开始", "停止", "停一下", "过来", "来一下", "看", "告诉")


@dataclass(frozen=True, slots=True)
class UtteranceTextAnalysis:
    transcript: str
    completeness_score: float
    direct_address_score: float
    self_talk_score: float


def analyze_utterance_text(transcript: str) -> UtteranceTextAnalysis:
    text = _normalize(transcript)
    if not text:
        return UtteranceTextAnalysis(transcript="", completeness_score=0.0, direct_address_score=0.0, self_talk_score=0.0)

    direct_hits = sum(1 for pattern in DIRECT_ADDRESS_PATTERNS if pattern in text)
    self_talk_hits = sum(1 for pattern in SELF_TALK_PATTERNS if pattern in text)
    is_question = any(marker in text for marker in QUESTION_MARKERS)
    is_command = any(marker in text for marker in COMMAND_MARKERS)
    is_filler = text in FILLER_PATTERNS or (len(text) <= 2 and all(char in FILLER_PATTERNS for char in text))

    direct_address_score = _clamp((direct_hits * 0.34) + (0.24 if is_question else 0.0) + (0.24 if is_command else 0.0))
    self_talk_score = _clamp((self_talk_hits * 0.34) + (0.32 if not direct_hits and not is_question and not is_command else 0.0))
    completeness_score = _completeness_score(text, is_filler, is_question, is_command, direct_hits)

    return UtteranceTextAnalysis(
        transcript=transcript.strip(),
        completeness_score=completeness_score,
        direct_address_score=direct_address_score,
        self_talk_score=self_talk_score,
    )


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"[\s,，。.!！;；:：、\"'“”‘’]+", "", lowered)


def _completeness_score(text: str, is_filler: bool, is_question: bool, is_command: bool, direct_hits: int) -> float:
    if is_filler:
        return 0.08
    if len(text) <= 1:
        return 0.12
    if len(text) <= 3 and direct_hits == 0 and not is_question and not is_command:
        return 0.35
    score = 0.56
    if len(text) >= 5:
        score += 0.18
    if is_question or is_command or direct_hits:
        score += 0.22
    return _clamp(score)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))