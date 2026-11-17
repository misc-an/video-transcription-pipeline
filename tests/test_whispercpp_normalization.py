"""Unit tests for whisper.cpp output normalization and CLI discovery."""

from video_transcription.backends.whispercpp_backend import (
    find_whisper_cli,
    normalize_whispercpp_output,
)


def test_normalize_whispercpp_output():
    """Verify whisper.cpp JSON output is normalized to the common dict format."""
    raw_json = {
        "result": {"language": "en"},
        "transcription": [
            {
                "timestamps": {"from": "00:00:00,000", "to": "00:00:01,500"},
                "offsets": {"from": 0, "to": 1500},
                "text": " Hello world.",
                "tokens": [
                    {
                        "text": "[_BEG_]",
                        "timestamps": {"from": "00:00:00,000", "to": "00:00:00,000"},
                        "offsets": {"from": 0, "to": 0},
                    },
                    {
                        "text": " Hello",
                        "timestamps": {"from": "00:00:00,050", "to": "00:00:00,450"},
                        "offsets": {"from": 50, "to": 450},
                    },
                    {
                        "text": " world.",
                        "timestamps": {"from": "00:00:00,500", "to": "00:00:01,500"},
                        "offsets": {"from": 500, "to": 1500},
                    },
                ],
            },
            {
                "timestamps": {"from": "00:00:02,000", "to": "00:00:03,500"},
                "offsets": {"from": 2000, "to": 3500},
                "text": "How are you?",
                "tokens": [
                    {
                        "text": "How",
                        "timestamps": {"from": "00:00:02,000", "to": "00:00:02,300"},
                        "offsets": {"from": 2000, "to": 2300},
                    },
                    {
                        "text": " are",
                        "timestamps": {"from": "00:00:02,350", "to": "00:00:02,700"},
                        "offsets": {"from": 2350, "to": 2700},
                    },
                    {
                        "text": " you?",
                        "timestamps": {"from": "00:00:02,750", "to": "00:00:03,500"},
                        "offsets": {"from": 2750, "to": 3500},
                    },
                ],
            },
        ],
    }

    result = normalize_whispercpp_output(raw_json)

    assert result["language"] == "en"
    assert result["text"] == "Hello world. How are you?"
    assert len(result["segments"]) == 2
    assert len(result["words"]) == 5

    # Check segment 0
    seg0 = result["segments"][0]
    assert seg0["start"] == 0.0
    assert seg0["end"] == 1.5
    assert seg0["text"] == "Hello world."
    assert seg0["speaker"] is None
    assert len(seg0["words"]) == 2

    # Check token normalization and special token filtering
    w0 = seg0["words"][0]
    assert w0["text"] == "Hello"
    assert w0["start"] == 0.05
    assert w0["end"] == 0.45
    assert w0["speaker"] is None

    w1 = seg0["words"][1]
    assert w1["text"] == "world."
    assert w1["start"] == 0.5
    assert w1["end"] == 1.5


def test_normalize_whispercpp_empty():
    """Verify normalization handles empty transcription cleanly."""
    raw_json = {"result": {"language": "fr"}, "transcription": []}
    result = normalize_whispercpp_output(raw_json)
    assert result["language"] == "fr"
    assert result["text"] == ""
    assert result["segments"] == []
    assert result["words"] == []
    assert result["speaker_turns"] == []


def test_find_whisper_cli():
    """Verify find_whisper_cli discovers the installed binary."""
    binary = find_whisper_cli()
    assert "whisper" in binary
