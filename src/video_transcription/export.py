"""Transcript formatting and export utilities for TXT, SRT, VTT, JSON, and InqScribe TSV."""

import json
from pathlib import Path
from typing import Any, Dict, List


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into standard SRT timestamp: HH:MM:SS,mmm with millisecond roll-over protection."""
    total_ms = max(0, int(round(seconds * 1000)))
    millis = total_ms % 1000
    total_secs = total_ms // 1000
    secs = total_secs % 60
    total_mins = total_secs // 60
    mins = total_mins % 60
    hours = total_mins // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm with millisecond roll-over protection."""
    total_ms = max(0, int(round(seconds * 1000)))
    millis = total_ms % 1000
    total_secs = total_ms // 1000
    secs = total_secs % 60
    total_mins = total_secs // 60
    mins = total_mins % 60
    hours = total_mins // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_inqscribe(seconds: float) -> str:
    """Format seconds into InqScribe timestamp: HH:MM:SS.ss with centisecond roll-over protection."""
    total_cs = max(0, int(round(seconds * 100 + 1e-9)))
    centis = total_cs % 100
    total_secs = total_cs // 100
    secs = total_secs % 60
    total_mins = total_secs // 60
    mins = total_mins % 60
    hours = total_mins // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{centis:02d}"


def save_transcription_outputs(
    result: Dict[str, Any],
    output_stem: Path,
    include_speakers: bool = True,
) -> Dict[str, Path]:
    """Save transcripts in TXT, SRT, VTT, TSV (InqScribe), and JSON formats without truncating dotted filenames."""
    segments: List[Dict[str, Any]] = result.get("segments", [])
    parent_dir = output_stem.parent
    stem_name = output_stem.name

    # 1. Plain Text (.txt) with timestamps and optional speaker markers
    txt_path = parent_dir / f"{stem_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker") if include_speakers else None
            start_val = seg.get("start")
            start_sec = 0.0 if start_val is None else float(start_val)
            end_val = seg.get("end")
            end_sec = start_sec if end_val is None else float(end_val)

            start_fmt = format_timestamp_vtt(start_sec)
            end_fmt = format_timestamp_vtt(end_sec)
            text = seg.get("text", "").strip()

            if speaker:
                f.write(f"[{start_fmt} - {end_fmt}] {speaker}: {text}\n\n")
            else:
                f.write(f"[{start_fmt} - {end_fmt}] {text}\n\n")

    # 2. SubRip Subtitles (.srt)
    srt_path = parent_dir / f"{stem_name}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_val = seg.get("start")
            start_sec = 0.0 if start_val is None else float(start_val)
            end_val = seg.get("end")
            end_sec = start_sec if end_val is None else float(end_val)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") if include_speakers else None

            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(start_sec)} --> {format_timestamp_srt(end_sec)}\n")
            if speaker:
                f.write(f"[{speaker}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")

    # 3. WebVTT (.vtt)
    vtt_path = parent_dir / f"{stem_name}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, start=1):
            start_val = seg.get("start")
            start_sec = 0.0 if start_val is None else float(start_val)
            end_val = seg.get("end")
            end_sec = start_sec if end_val is None else float(end_val)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") if include_speakers else None

            f.write(f"{i}\n")
            f.write(f"{format_timestamp_vtt(start_sec)} --> {format_timestamp_vtt(end_sec)}\n")
            if speaker:
                f.write(f"<v {speaker}>{text}</v>\n\n")
            else:
                f.write(f"{text}\n\n")

    # 4. InqScribe Tab-Delimited (.tsv)
    tsv_path = parent_dir / f"{stem_name}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        for seg in segments:
            start_val = seg.get("start")
            start_sec = 0.0 if start_val is None else float(start_val)
            end_val = seg.get("end")
            end_sec = start_sec if end_val is None else float(end_val)

            start_fmt = format_timestamp_inqscribe(start_sec)
            end_fmt = format_timestamp_inqscribe(end_sec)

            # Sanitize text: collapse newlines and tabs to single spaces
            raw_text = seg.get("text", "").strip()
            clean_text = " ".join(raw_text.split())

            if include_speakers:
                speaker = seg.get("speaker") or ""
                f.write(f"{start_fmt}\t{end_fmt}\t{speaker}\t{clean_text}\n")
            else:
                f.write(f"{start_fmt}\t{end_fmt}\t{clean_text}\n")

    # 5. Structured JSON (.json)
    json_path = parent_dir / f"{stem_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return {
        "txt": txt_path,
        "srt": srt_path,
        "vtt": vtt_path,
        "tsv": tsv_path,
        "json": json_path,
    }
