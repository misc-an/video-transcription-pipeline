"""Main transcription and diarization pipeline execution with pluggable ASR backends."""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Suppress harmless third-party runtime warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote.*")
warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*upgrade_checkpoint.*")

from .audio import extract_audio_ffmpeg, get_media_duration_ffprobe
from .export import format_timestamp_vtt, save_transcription_outputs


def is_cjk(char: str) -> bool:
    """Check if character is within Chinese/Japanese/Korean unicode ranges."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def smart_join_tokens(tokens: List[str]) -> str:
    """Join tokens intelligently: insert spaces between non-CJK words, no spaces between CJK chars."""
    if not tokens:
        return ""
    result = []
    for i, token in enumerate(tokens):
        if i == 0:
            result.append(token)
            continue
        prev = tokens[i - 1]
        prev_is_cjk = prev and is_cjk(prev[-1])
        curr_is_cjk = token and is_cjk(token[0])
        # Only omit space when transitioning between CJK characters
        if prev_is_cjk and curr_is_cjk:
            result.append(token)
        else:
            result.append(" " + token)
    return "".join(result).strip()


def smooth_speaker_assignments(
    words: List[Dict[str, Any]],
    min_words_threshold: int = 3,
    pause_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """Smooth isolated speaker turn glitches and boundary artifacts in continuous speech."""
    if len(words) < 3:
        return words

    # Pass 1a: Remove isolated 1-word speaker blips surrounded by the same speaker
    for i in range(1, len(words) - 1):
        prev_w = words[i - 1]
        curr_w = words[i]
        next_w = words[i + 1]

        if (
            prev_w.get("speaker")
            and prev_w["speaker"] == next_w.get("speaker")
            and curr_w.get("speaker") != prev_w["speaker"]
        ):
            pause1 = float(curr_w["start"]) - float(prev_w["end"])
            pause2 = float(next_w["start"]) - float(curr_w["end"])
            if pause1 < pause_threshold and pause2 < pause_threshold:
                curr_w["speaker"] = prev_w["speaker"]

    # Pass 1b: Remove isolated 2-word speaker blips surrounded by the same speaker
    for i in range(1, len(words) - 2):
        prev_w = words[i - 1]
        w1 = words[i]
        w2 = words[i + 1]
        next_w = words[i + 2]

        if (
            prev_w.get("speaker")
            and prev_w["speaker"] == next_w.get("speaker")
            and w1.get("speaker") != prev_w["speaker"]
            and w2.get("speaker") != prev_w["speaker"]
        ):
            pause1 = float(w1["start"]) - float(prev_w["end"])
            pause2 = float(w2["start"]) - float(w1["end"])
            pause3 = float(next_w["start"]) - float(w2["end"])
            if pause1 < pause_threshold and pause2 < pause_threshold and pause3 < pause_threshold:
                w1["speaker"] = prev_w["speaker"]
                w2["speaker"] = prev_w["speaker"]

    # Pass 2: Boundary smoothing for 1-2 word blips adjacent to a long continuous run without a pause
    runs: List[Tuple[int, int, Optional[str]]] = []
    i = 0
    while i < len(words):
        spk = words[i].get("speaker")
        start_idx = i
        while i < len(words) and words[i].get("speaker") == spk:
            i += 1
        runs.append((start_idx, i, spk))

    for r_idx in range(len(runs)):
        start_idx, end_idx, spk = runs[r_idx]
        run_len = end_idx - start_idx

        if run_len < min_words_threshold:
            # Check merge forward into next run if next run is long and there is no pause
            if r_idx + 1 < len(runs):
                next_start, next_end, next_spk = runs[r_idx + 1]
                next_len = next_end - next_start
                pause_after = float(words[next_start]["start"]) - float(words[end_idx - 1]["end"])
                if pause_after < pause_threshold and next_len >= min_words_threshold and next_spk is not None:
                    for j in range(start_idx, end_idx):
                        words[j]["speaker"] = next_spk
                    continue

            # Check merge backward into previous run if previous run was long and there is no pause
            if r_idx - 1 >= 0:
                prev_start, prev_end, prev_spk = runs[r_idx - 1]
                prev_len = prev_end - prev_start
                pause_before = float(words[start_idx]["start"]) - float(words[prev_end - 1]["end"])
                prev_ends_sentence = words[prev_end - 1]["text"][-1:] in ".?!"
                if pause_before < pause_threshold and not prev_ends_sentence and prev_len >= min_words_threshold and prev_spk is not None:
                    for j in range(start_idx, end_idx):
                        words[j]["speaker"] = prev_spk

    return words


def assign_speakers_to_words(
    words: List[Dict[str, Any]],
    speaker_turns: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Assign speaker labels to words based on maximum temporal overlap with diarization intervals."""
    if not speaker_turns:
        return words

    # Filter out micro-turns (< 0.2s) that are acoustic noise or echo artifacts
    valid_turns = [t for t in speaker_turns if (t[1] - t[0]) >= 0.2] or speaker_turns

    for w in words:
        w_start = float(w["start"])
        w_end = float(w["end"])
        best_overlap = 0.0
        best_speaker = None
        min_dist = float("inf")
        closest_speaker = None

        w_mid = (w_start + w_end) / 2.0

        for t_start, t_end, speaker in valid_turns:
            overlap = max(0.0, min(w_end, t_end) - max(w_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

            dist = 0.0 if (t_start <= w_mid <= t_end) else min(abs(w_mid - t_start), abs(w_mid - t_end))
            if dist < min_dist:
                min_dist = dist
                closest_speaker = speaker

        w["speaker"] = best_speaker if best_speaker is not None else closest_speaker

    return smooth_speaker_assignments(words)


def group_words_into_segments(
    words: List[Dict[str, Any]],
    max_duration: float = 7.0,
    pause_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """Group aligned words into dialogue segments breaking on speaker change, pauses, or sentence punctuation."""
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []
    current_speaker = words[0].get("speaker")

    for w in words:
        speaker = w.get("speaker")
        speaker_changed = (speaker != current_speaker) if speaker is not None else False
        pause = (float(w["start"]) - float(current_words[-1]["end"])) if current_words else 0.0
        dur = (float(w["end"]) - float(current_words[0]["start"])) if current_words else 0.0
        ends_sentence = (
            current_words and current_words[-1]["text"].rstrip()[-1:] in ".?!。？！"
        )

        if current_words and (speaker_changed or pause > pause_threshold or dur > max_duration or ends_sentence):
            token_texts = [x["text"] for x in current_words]
            seg_text = smart_join_tokens(token_texts)
            segments.append({
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "text": seg_text,
                "speaker": current_speaker,
                "words": current_words,
            })
            current_words = []
            current_speaker = speaker

        current_words.append(w)

    if current_words:
        token_texts = [x["text"] for x in current_words]
        seg_text = smart_join_tokens(token_texts)
        segments.append({
            "start": float(current_words[0]["start"]),
            "end": float(current_words[-1]["end"]),
            "text": seg_text,
            "speaker": current_speaker,
            "words": current_words,
        })

    return segments


def determine_output_dir(media_path: Path, output_dir: Optional[str] = None) -> Path:
    """Determine destination output directory.

    1. If explicit output_dir is given, use it.
    2. If media_path is located inside a 'data/input' tree, automatically mirror to 'data/output'.
    3. If peer or parent 'output' directory exists alongside an 'input' directory, mirror to it.
    4. If './data/output' exists in the current working directory, route there.
    5. Fallback to media_path.parent.
    """
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    media_resolved = media_path.resolve()
    parts = list(media_resolved.parts)
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "input" and i > 0 and parts[i - 1] == "data":
            new_parts = parts[:i] + ["output"] + parts[i + 1 : -1]
            return Path(*new_parts)
        elif parts[i] == "input":
            peer_output = Path(*parts[:i]) / "output"
            if peer_output.exists():
                sub_parts = parts[i + 1 : -1]
                return peer_output.joinpath(*sub_parts)

    cwd_output = Path.cwd() / "data" / "output"
    if cwd_output.exists():
        return cwd_output.resolve()

    return media_resolved.parent


def run_pipeline(
    media_file: str,
    output_dir: Optional[str] = None,
    engine: Optional[str] = None,
    # Qwen-specific
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    # whisper.cpp-specific
    whisper_model: str = "large-v3",
    whisper_threads: Optional[int] = None,
    no_gpu: bool = False,
    # Shared
    language: Optional[str] = None,
    device: Optional[str] = None,
    do_align: bool = True,
    do_diarize: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    start_time: Optional[str] = None,
    duration: Optional[float] = None,
    chunk_size: float = 30.0,
    timestamps_in_txt: bool = False,
    cleanup_audio: bool = False,
):
    media_path = Path(media_file).expanduser().resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Input media file not found: {media_path}")

    # Resolve ASR engine (normalizing aliases)
    from .backends import get_available_backends, get_default_backend

    if engine:
        norm_engine = engine.lower()
        if norm_engine in ("whisper", "whisper.cpp", "whispercpp"):
            norm_engine = "whispercpp"

        available = get_available_backends()
        if norm_engine not in available:
            raise ImportError(
                f"ASR engine '{engine}' is not available.\n"
                f"Available backends: {available or '(none)'}.\n"
                f"To install:\n"
                f"  • Qwen3-ASR:   uv sync --extra qwen\n"
                f"  • whisper.cpp: brew install whisper-cpp"
            )
        chosen_engine = norm_engine
    else:
        chosen_engine = get_default_backend()

    # Set up output directory and base stem preserving internal dots
    out_dir = determine_output_dir(media_path, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = media_path.stem
    if base_name.endswith(".extracted"):
        base_name = base_name[:-len(".extracted")]
    output_stem = out_dir / base_name

    # Determine hardware device and dtype for PyTorch stages (Qwen & PyAnnote)
    import torch
    if device is None:
        if torch.cuda.is_available():
            chosen_device = "cuda"
            chosen_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            chosen_device = "mps"
            chosen_dtype = torch.float16
        else:
            chosen_device = "cpu"
            chosen_dtype = torch.float32
    else:
        chosen_device = device
        if chosen_device.startswith("cuda"):
            chosen_dtype = torch.bfloat16
        elif chosen_device == "mps":
            chosen_dtype = torch.float16
        else:
            chosen_dtype = torch.float32

    console = Console()

    # Build engine display info
    if chosen_engine == "qwen":
        engine_display = f"Qwen3-ASR ({asr_model_name})"
        device_display = f"{chosen_device.upper()}, {chosen_dtype}"
    else:
        gpu_label = "CPU only" if no_gpu else "Metal GPU (Apple Silicon)"
        engine_display = f"whisper.cpp ({whisper_model})"
        device_display = f"ASR: {gpu_label}, Diarization: {chosen_device.upper()}"

    # Display configuration panel
    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Key", style="bold cyan", width=18)
    info_table.add_column("Value", style="white")
    info_table.add_row("Input Media", str(media_path))
    info_table.add_row("Output Dir", str(out_dir))
    info_table.add_row("ASR Engine", engine_display)
    info_table.add_row("Device", device_display)
    if chosen_engine == "qwen":
        info_table.add_row("Chunk Size", f"{chunk_size}s")
        info_table.add_row("Forced Alignment", "Enabled" if do_align else "Disabled")
    else:
        info_table.add_row("Word Timestamps", "Enabled (Attention DTW)")
    info_table.add_row("Diarization", "Enabled (PyAnnote)" if do_diarize else "Disabled")
    if duration:
        info_table.add_row("Segment Limit", f"start={start_time or 0}s, duration={duration}s")
    console.print()
    console.print(Panel(info_table, title="[bold cyan]Video-to-Text Transcription Pipeline[/bold cyan]", border_style="cyan"))
    console.print()

    # Suppress verbose third-party logging to prevent terminal line duplicates
    try:
        import transformers
        transformers.utils.logging.disable_progress_bar()
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass
    import huggingface_hub
    huggingface_hub.utils.disable_progress_bars()

    # Single sleek, animated progress bar tracking pipeline stages and elapsed time
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("[dim]•[/dim]"),
        TextColumn("[dim]Elapsed:[/dim]"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task("[1/5] Extracting audio (FFmpeg)...", total=5.0)

        # Stage 1: Audio Extraction (FFmpeg)
        extracted_wav = out_dir / f"{base_name}.extracted.wav"
        if media_path.resolve() == extracted_wav.resolve() and start_time is None and duration is None:
            pass
        else:
            if media_path.resolve() == extracted_wav.resolve():
                temp_wav = out_dir / f"{base_name}.tmp.wav"
                extract_audio_ffmpeg(
                    media_path=media_path,
                    output_wav_path=temp_wav,
                    start_time=start_time,
                    duration=duration,
                )
                temp_wav.replace(extracted_wav)
            else:
                extract_audio_ffmpeg(
                    media_path=media_path,
                    output_wav_path=extracted_wav,
                    start_time=start_time,
                    duration=duration,
                )

        # Detect incomplete or truncated media files on disk
        if duration is None:
            try:
                import soundfile as sf
                wav_info = sf.info(str(extracted_wav))
                actual_wav_sec = wav_info.duration
                container_sec = get_media_duration_ffprobe(media_path)
                if container_sec and container_sec > actual_wav_sec + 15.0:
                    console.print(
                        f"\n[bold yellow]⚠ Warning: Input media file is truncated on disk![/bold yellow]\n"
                        f"  • Container header indicates: [cyan]{format_timestamp_vtt(container_sec)}[/cyan] ({container_sec/60:.1f} min)\n"
                        f"  • Actual audio stream on disk: [cyan]{format_timestamp_vtt(actual_wav_sec)}[/cyan] ({actual_wav_sec/60:.1f} min)\n"
                        f"  • File size on disk: {media_path.stat().st_size / (1024*1024):.1f} MB (an intact {container_sec/60:.1f} min file would be ~{container_sec / actual_wav_sec * (media_path.stat().st_size / (1024*1024)):.0f} MB)\n"
                        f"  • The media stream was interrupted during copying or download. Processing available audio.\n"
                    )
            except Exception:
                pass

        # Stage 2-3: ASR Transcription & Alignment (dispatched to selected backend)
        if chosen_engine == "qwen":
            from .backends.qwen_backend import transcribe_qwen
            pipeline_result = transcribe_qwen(
                audio_path=str(extracted_wav),
                asr_model_name=asr_model_name,
                aligner_model_name=aligner_model_name,
                language=language,
                device=chosen_device,
                dtype=chosen_dtype,
                do_align=do_align,
                chunk_size=chunk_size,
                progress=progress,
                task_id=task_id,
            )
        elif chosen_engine == "whispercpp":
            from .backends.whispercpp_backend import transcribe_whispercpp
            pipeline_result = transcribe_whispercpp(
                audio_path=str(extracted_wav),
                model_name=whisper_model,
                language=language,
                threads=whisper_threads,
                no_gpu=no_gpu,
                progress=progress,
                task_id=task_id,
            )
        else:
            raise ValueError(f"Unknown ASR engine: {chosen_engine}")

        full_text = pipeline_result["text"]
        detected_lang = pipeline_result["language"]
        words = pipeline_result["words"]
        segments = pipeline_result["segments"]

        # Stage 4: Speaker Diarization (PyAnnote) — shared between backends
        speaker_labeled = False
        speaker_turns: List[Tuple[float, float, str]] = []
        if do_diarize:
            progress.update(task_id, description="[4/5] Diarizing speakers (PyAnnote)...")
            token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or huggingface_hub.get_token()

            if not token:
                console.print("[yellow]⚠ Hugging Face token missing for speaker diarization.[/yellow]")
            else:
                try:
                    import torchaudio
                    from pyannote.audio import Pipeline

                    diarize_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=token,
                    )
                    diarize_pipeline.to(torch.device(chosen_device))

                    waveform, sample_rate = torchaudio.load(str(extracted_wav))
                    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers

                    with torch.inference_mode():
                        diarization_output = diarize_pipeline(audio_input, **diarize_kwargs)
                    annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        speaker_turns.append((turn.start, turn.end, speaker))

                    if words and speaker_turns:
                        assign_speakers_to_words(words, speaker_turns)
                        speaker_labeled = True
                        # Re-group segments with speaker labels
                        segments = group_words_into_segments(words)
                    elif speaker_turns:
                        for seg in segments:
                            best_spk = None
                            best_overlap = 0.0
                            for t_start, t_end, spk in speaker_turns:
                                overlap = max(0.0, min(seg["end"], t_end) - max(seg["start"], t_start))
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_spk = spk
                            seg["speaker"] = best_spk
                        speaker_labeled = True
                except Exception as e:
                    console.print(f"[yellow]⚠ Speaker diarization failed: {e}[/yellow]")
        else:
            progress.update(task_id, description="[4/5] Diarization skipped...")
        progress.update(task_id, completed=4.0)

        # Rebuild pipeline_result with diarization data
        pipeline_result = {
            "text": full_text,
            "language": detected_lang,
            "segments": segments,
            "words": words,
            "speaker_turns": [
                {"start": t[0], "end": t[1], "speaker": t[2]} for t in speaker_turns
            ],
        }

        # Stage 5: Exporting outputs
        progress.update(task_id, description="[5/5] Exporting transcripts...")
        saved_files = save_transcription_outputs(
            result=pipeline_result,
            output_stem=output_stem,
            include_speakers=speaker_labeled,
            timestamps_in_txt=timestamps_in_txt,
        )
        progress.update(task_id, completed=5.0, description="[bold green]Pipeline completed successfully![/bold green]")

    # Optional Audio Cleanup
    if cleanup_audio and extracted_wav.exists():
        extracted_wav.unlink()

    # Display final results panel
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("Type", style="bold cyan", width=18)
    summary_table.add_column("File", style="green")
    summary_table.add_row("Engine", chosen_engine.title())
    summary_table.add_row("Detected Language", detected_lang)
    summary_table.add_row("TXT Transcript", str(saved_files["txt"]))
    summary_table.add_row("SRT Subtitles", str(saved_files["srt"]))
    summary_table.add_row("VTT Subtitles", str(saved_files["vtt"]))
    summary_table.add_row("TSV InqScribe", str(saved_files["tsv"]))
    summary_table.add_row("JSON Structure", str(saved_files["json"]))

    console.print()
    console.print(Panel(summary_table, title="[bold green]✓ Pipeline Completed Successfully[/bold green]", border_style="green"))
    console.print()
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Universal Video-to-Text Pipeline with Qwen3-ASR and whisper.cpp backends, FFmpeg, and PyAnnote Speaker Diarization."
    )
    parser.add_argument("media_file", type=str, help="Path to input video or audio file.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save generated transcripts and audio (default: next to media file).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["qwen", "whispercpp", "whisper", "whisper.cpp"],
        help="ASR engine: 'qwen' (Qwen3-ASR, default) or 'whispercpp' (whisper.cpp on Metal GPU).",
    )

    # Qwen-specific options
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen ASR model repository ID (default: Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument(
        "--aligner-model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Qwen Forced Aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B).",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=30.0,
        help="Chunk duration in seconds for Qwen3 ASR and forced alignment (default: 30.0s).",
    )

    # whisper.cpp-specific options
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="whisper.cpp GGML model name or path (default: large-v3; e.g. base, large-v3-q5_0).",
    )
    parser.add_argument(
        "--whisper-threads",
        type=int,
        default=None,
        help="Number of threads for whisper.cpp (default: auto).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU/Metal acceleration in whisper.cpp and run on CPU.",
    )

    # Shared options
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language (e.g. English, Chinese, or code en, zh). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-align",
        dest="do_align",
        action="store_false",
        help="Disable word-level forced alignment step (Qwen engine).",
    )
    parser.set_defaults(do_align=True)

    parser.add_argument(
        "--diarize",
        dest="do_diarize",
        action="store_true",
        help="Enable Speaker Diarization (PyAnnote).",
    )
    parser.add_argument(
        "--no-diarize",
        dest="do_diarize",
        action="store_false",
        help="Disable Speaker Diarization.",
    )
    parser.set_defaults(do_diarize=False)

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for PyAnnote speaker diarization.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (optional).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (optional).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start offset for media processing (e.g. '0', '00:01:30').",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to process (useful for quick testing, e.g. 10).",
    )
    parser.add_argument(
        "--timestamps-in-txt",
        action="store_true",
        help="Include [HH:MM:SS.mmm - HH:MM:SS.mmm] timestamps in the plain .txt export.",
    )
    parser.add_argument(
        "--cleanup-audio",
        action="store_true",
        help="Delete intermediate 16kHz WAV file after completion.",
    )

    args = parser.parse_args()

    # Normalize user model flag (e.g. "1.7b" -> "Qwen/Qwen3-ASR-1.7B")
    model_name = args.model
    if model_name.lower() in ["1.7b", "1.7", "qwen3-asr-1.7b"]:
        model_name = "Qwen/Qwen3-ASR-1.7B"

    # If --hf-token is provided or HF_TOKEN is in environment, enable diarization by default unless explicitly --no-diarize
    if (args.hf_token or os.environ.get("HF_TOKEN")) and "--no-diarize" not in sys.argv:
        args.do_diarize = True

    run_pipeline(
        media_file=args.media_file,
        output_dir=args.output_dir,
        engine=args.engine,
        asr_model_name=model_name,
        aligner_model_name=args.aligner_model,
        whisper_model=args.whisper_model,
        whisper_threads=args.whisper_threads,
        no_gpu=args.no_gpu,
        language=args.language,
        device=args.device,
        do_align=args.do_align,
        do_diarize=args.do_diarize,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        start_time=args.start_time,
        duration=args.duration,
        chunk_size=args.chunk_size,
        timestamps_in_txt=args.timestamps_in_txt,
        cleanup_audio=args.cleanup_audio,
    )


if __name__ == "__main__":
    main()
