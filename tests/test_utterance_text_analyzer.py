from loona_wakeup.engine.utterance_text_analyzer import analyze_utterance_text


def test_analyzer_scores_direct_address_command() -> None:
    analysis = analyze_utterance_text("露娜，帮我打开灯")
    assert analysis.completeness_score >= 0.70
    assert analysis.direct_address_score >= 0.50
    assert analysis.self_talk_score < 0.40


def test_analyzer_scores_self_talk() -> None:
    analysis = analyze_utterance_text("我想想这个应该怎么做")
    assert analysis.completeness_score >= 0.70
    assert analysis.self_talk_score > 0.62
    assert analysis.direct_address_score < 0.30


def test_analyzer_reduces_filler_completeness() -> None:
    analysis = analyze_utterance_text("嗯")
    assert analysis.completeness_score < 0.45
    assert analysis.direct_address_score == 0.0