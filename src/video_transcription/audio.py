"""Audio extraction and processing utilities using FFmpeg."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_ffmpeg() -> str:
    """Locate ffmpeg binary on the system."""
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "FFmpeg binary not found. Please install ffmpeg (e.g. 'brew install ffmpeg') or add it to PATH."
    )


def get_media_duration_ffprobe(media_path: Path) -> Optional[float]:
    """Retrieve media container duration in seconds via ffprobe if available."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        for candidate in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe"]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                ffprobe_bin = candidate
                break
    if not ffprobe_bin:
        return None
    try:
        cmd = [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(media_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            import json
            data = json.loads(res.stdout)
            val = data.get("format", {}).get("duration")
            return float(val) if val is not None else None
    except Exception:
        pass
    return None


def extract_audio_ffmpeg(
    media_path: Path,
    output_wav_path: Path,
    start_time: Optional[str] = None,
    duration: Optional[float] = None,
) -> Path:
    """
    Universal media to 16kHz mono 16-bit PCM WAV extraction using FFmpeg.
    Supports any video (.webm, .mp4, .mkv, .mov, .avi, etc.) or audio (.mp3, .m4a, .flac, etc.).
    """
    ffmpeg_bin = find_ffmpeg()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg_bin, "-y"]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    cmd.extend(["-i", str(media_path)])

    if duration is not None:
        cmd.extend(["-t", str(duration)])

    # Stream-agnostic audio extraction: no video, 16kHz, mono, 16-bit PCM
    cmd.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path),
    ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr_data = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed (code {process.returncode}):\n{stderr_data}")

    return output_wav_path
