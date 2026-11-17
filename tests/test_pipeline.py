"""Automated unit tests for pipeline components, backends, and export utilities."""

import json
from pathlib import Path
import pytest

from video_transcription.backends import get_available_backends, get_default_backend
from video_transcription.export import (
    format_timestamp_inqscribe,
    format_timestamp_srt,
    format_timestamp_vtt,
    save_transcription_outputs,
)
from video_transcription.pipeline import (
    assign_speakers_to_words,
    determine_output_dir,
    group_words_into_segments,
    smart_join_tokens,
    smooth_speaker_assignments,
)


def test_format_timestamp_srt():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(12.345) == "00:00:12,345"
    # Millisecond overflow roll-over protection
    assert format_timestamp_srt(1.9999) == "00:00:02,000"
    assert format_timestamp_srt(59.9996) == "00:01:00,000"
    assert format_timestamp_srt(3599.9999) == "01:00:00,000"


def test_format_timestamp_vtt():
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(12.345) == "00:00:12.345"
    assert format_timestamp_vtt(1.9999) == "00:00:02.000"
    assert format_timestamp_vtt(59.9996) == "00:01:00.000"


def test_format_timestamp_inqscribe():
    assert format_timestamp_inqscribe(0.0) == "00:00:00.00"
    assert format_timestamp_inqscribe(12.345) == "00:00:12.35"
    assert format_timestamp_inqscribe(1.9999) == "00:00:02.00"
    assert format_timestamp_inqscribe(59.996) == "00:01:00.00"
    assert format_timestamp_inqscribe(3599.996) == "01:00:00.00"


def test_smart_join_tokens():
    assert smart_join_tokens(["Hello", "world"]) == "Hello world"
    assert smart_join_tokens(["你", "好", "世", "界"]) == "你好世界"
    assert smart_join_tokens(["Hello", "世界"]) == "Hello 世界"
    assert smart_join_tokens([]) == ""


def test_assign_speakers_to_words():
    words = [
        {"text": "Good", "start": 0.1, "end": 0.5, "speaker": None},
        {"text": "morning", "start": 0.6, "end": 1.0, "speaker": None},
        {"text": "Thank", "start": 2.1, "end": 2.5, "speaker": None},
        {"text": "you", "start": 2.6, "end": 2.9, "speaker": None},
    ]
    turns = [
        (0.0, 1.5, "SPEAKER_00"),
        (2.0, 3.5, "SPEAKER_01"),
    ]
    assigned = assign_speakers_to_words(words, turns)
    assert assigned[0]["speaker"] == "SPEAKER_00"
    assert assigned[1]["speaker"] == "SPEAKER_00"
    assert assigned[2]["speaker"] == "SPEAKER_01"
    assert assigned[3]["speaker"] == "SPEAKER_01"


def test_assign_speakers_filters_micro_turns():
    """Verify acoustic noise spikes shorter than 0.2s are filtered out."""
    words = [
        {"text": "Welcome", "start": 0.1, "end": 0.5, "speaker": None},
        {"text": "everyone", "start": 0.6, "end": 1.0, "speaker": None},
    ]
    turns = [
        (0.2, 0.3, "SPEAKER_NOISE"),  # 0.1s turn should be filtered
        (0.0, 1.5, "SPEAKER_00"),
    ]
    assigned = assign_speakers_to_words(words, turns)
    assert assigned[0]["speaker"] == "SPEAKER_00"
    assert assigned[1]["speaker"] == "SPEAKER_00"


def test_smooth_speaker_assignments_surrounded_blip():
    """Verify isolated 1- and 2-word speaker blips are smoothed to surrounding speaker."""
    words = [
        {"text": "I", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"},
        {"text": "think", "start": 0.25, "end": 0.5, "speaker": "SPEAKER_01"},  # 1-word blip
        {"text": "that", "start": 0.55, "end": 0.8, "speaker": "SPEAKER_00"},
        {"text": "is", "start": 0.85, "end": 1.0, "speaker": "SPEAKER_00"},
        {"text": "very", "start": 1.05, "end": 1.25, "speaker": "SPEAKER_02"},  # 2-word blip
        {"text": "good", "start": 1.30, "end": 1.50, "speaker": "SPEAKER_02"},
        {"text": "indeed", "start": 1.55, "end": 1.80, "speaker": "SPEAKER_00"},
    ]
    smoothed = smooth_speaker_assignments(words)
    speakers = [w["speaker"] for w in smoothed]
    assert speakers == ["SPEAKER_00"] * 7


def test_smooth_speaker_assignments_boundary_merge():
    """Verify 1-2 word boundary flip is merged into continuous long utterance."""
    words = [
        {"text": "okay", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_01"},
        # 1.0s pause between turns
        {"text": "so", "start": 1.4, "end": 1.6, "speaker": "SPEAKER_00"},
        {"text": "I'd", "start": 1.65, "end": 1.85, "speaker": "SPEAKER_00"},
        # Continuous speech into SPEAKER_02 without pause (< 0.4s)
        {"text": "like", "start": 1.90, "end": 2.1, "speaker": "SPEAKER_02"},
        {"text": "to", "start": 2.15, "end": 2.3, "speaker": "SPEAKER_02"},
        {"text": "thank", "start": 2.35, "end": 2.6, "speaker": "SPEAKER_02"},
        {"text": "you", "start": 2.65, "end": 2.8, "speaker": "SPEAKER_02"},
    ]
    smoothed = smooth_speaker_assignments(words)
    assert smoothed[0]["speaker"] == "SPEAKER_01"
    # 'so I'd' merged into SPEAKER_02
    assert [w["speaker"] for w in smoothed[1:]] == ["SPEAKER_02"] * 6


def test_group_words_into_segments():
    words = [
        {"text": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
        {"text": "world.", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
        {"text": "How", "start": 2.0, "end": 2.4, "speaker": "SPEAKER_01"},
        {"text": "are", "start": 2.5, "end": 2.8, "speaker": "SPEAKER_01"},
        {"text": "you?", "start": 2.9, "end": 3.2, "speaker": "SPEAKER_01"},
    ]
    segs = group_words_into_segments(words, max_duration=10.0, pause_threshold=1.0)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 0.9

    assert segs[1]["speaker"] == "SPEAKER_01"
    assert segs[1]["text"] == "How are you?"
    assert segs[1]["start"] == 2.0
    assert segs[1]["end"] == 3.2


def test_save_transcription_outputs(tmp_path: Path):
    result = {
        "text": "Hello world.",
        "language": "English",
        "segments": [
            {
                "start": None,  # Test None timestamp protection
                "end": 1.5,
                "text": "Hello world.",
                "speaker": "SPEAKER_00",
            }
        ],
    }

    # Test filename with dots
    output_stem = tmp_path / "interview.2026.10.file"
    outputs = save_transcription_outputs(result, output_stem, include_speakers=True)

    # Check paths exist and preserve full stem
    assert outputs["txt"].name == "interview.2026.10.file.txt"
    assert outputs["srt"].name == "interview.2026.10.file.srt"
    assert outputs["vtt"].name == "interview.2026.10.file.vtt"
    assert outputs["tsv"].name == "interview.2026.10.file.tsv"
    assert outputs["json"].name == "interview.2026.10.file.json"

    assert outputs["txt"].exists()
    assert outputs["srt"].exists()
    assert outputs["vtt"].exists()
    assert outputs["tsv"].exists()
    assert outputs["json"].exists()

    # Verify content
    txt_content = outputs["txt"].read_text(encoding="utf-8")
    assert txt_content == "SPEAKER_00: Hello world.\n\n"

    # Verify content
    vtt_content = outputs["vtt"].read_text(encoding="utf-8")
    assert "WEBVTT" in vtt_content
    assert "<v SPEAKER_00>Hello world.</v>" in vtt_content

    srt_content = outputs["srt"].read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in srt_content
    assert "[SPEAKER_00] Hello world." in srt_content

    tsv_content = outputs["tsv"].read_text(encoding="utf-8")
    assert tsv_content == "00:00:00.00\t00:00:01.50\tSPEAKER_00\tHello world.\n"

    json_content = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert json_content["language"] == "English"

    # Test without speakers (3-column TSV)
    outputs_no_spk = save_transcription_outputs(result, tmp_path / "no_spk", include_speakers=False)
    tsv_no_spk_content = outputs_no_spk["tsv"].read_text(encoding="utf-8")
    assert tsv_no_spk_content == "00:00:00.00\t00:00:01.50\tHello world.\n"


def test_determine_output_dir(tmp_path: Path):
    # 1. Explicit output dir
    custom_dir = tmp_path / "custom"
    assert determine_output_dir(tmp_path / "video.mp4", str(custom_dir)) == custom_dir.resolve()

    # 2. Inside data/input tree
    input_file = tmp_path / "data" / "input" / "workshop" / "2" / "part1.mp4"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.touch()

    expected_output = tmp_path / "data" / "output" / "workshop" / "2"
    assert determine_output_dir(input_file) == expected_output.resolve()

    # 3. Inside top-level input file
    input_top = tmp_path / "data" / "input" / "sample.mp4"
    input_top.touch()
    assert determine_output_dir(input_top) == (tmp_path / "data" / "output").resolve()


def test_get_available_backends():
    backends = get_available_backends()
    assert isinstance(backends, list)
    # The environment has at least one backend installed
    assert len(backends) >= 1
    assert any(b in ("qwen", "whispercpp") for b in backends)


def test_get_default_backend():
    default = get_default_backend()
    assert default in ("qwen", "whispercpp")
