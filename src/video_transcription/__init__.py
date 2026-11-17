"""
Universal Video-to-Text Transcription & Speaker Diarization Pipeline.
Powered by FFmpeg, Qwen3-ASR (1.7B), whisper.cpp (Metal GPU), and PyAnnote.
"""

from .audio import extract_audio_ffmpeg, find_ffmpeg, get_media_duration_ffprobe
from .backends import get_available_backends, get_default_backend
from .export import (
    format_timestamp_inqscribe,
    format_timestamp_srt,
    format_timestamp_vtt,
    save_transcription_outputs,
)
from .pipeline import (
    assign_speakers_to_words,
    determine_output_dir,
    group_words_into_segments,
    main as transcribe_main,
    run_pipeline,
)
from .weights import (
    download_diarization_weights,
    download_qwen_weights,
    download_whispercpp_weights,
)

__version__ = "0.4.0"
__all__ = [
    "assign_speakers_to_words",
    "determine_output_dir",
    "download_diarization_weights",
    "download_qwen_weights",
    "download_whispercpp_weights",
    "extract_audio_ffmpeg",
    "find_ffmpeg",
    "format_timestamp_inqscribe",
    "format_timestamp_srt",
    "format_timestamp_vtt",
    "get_available_backends",
    "get_default_backend",
    "group_words_into_segments",
    "run_pipeline",
    "save_transcription_outputs",
    "transcribe_main",
]
