"""whisper.cpp backend: fast, native Apple Silicon Metal/CPU ASR using whisper-cli."""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.progress import Progress, TaskID


def is_cjk_char(char: str) -> bool:
    """Check if character is within Chinese/Japanese/Korean unicode ranges."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def find_whisper_cli() -> str:
    """Locate whisper-cli binary on the system."""
    binary = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    if binary:
        return binary

    candidate_paths = [
        "/opt/homebrew/bin/whisper-cli",
        "/usr/local/bin/whisper-cli",
        "/usr/bin/whisper-cli",
    ]
    # Check Homebrew Cellar if unlinked
    candidate_paths.extend(glob.glob("/opt/homebrew/Cellar/whisper.cpp/*/bin/whisper-cli"))
    candidate_paths.extend(glob.glob("/usr/local/Cellar/whisper.cpp/*/bin/whisper-cli"))

    for candidate in candidate_paths:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "whisper.cpp binary ('whisper-cli') not found on system.\n"
        "Please install it:\n"
        "  • macOS: brew install whisper-cpp\n"
        "  • Linux: build from https://github.com/ggml-org/whisper.cpp or install package"
    )


def resolve_whispercpp_model(model_name_or_path: str) -> str:
    """Resolve model to a local file path, downloading from Hugging Face if needed."""
    candidate = Path(model_name_or_path).expanduser().resolve()
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    # Normalize name to ggml filename
    name = model_name_or_path.strip()
    if not name.startswith("ggml-"):
        name = f"ggml-{name}"
    if not name.endswith(".bin"):
        name = f"{name}.bin"

    from huggingface_hub import hf_hub_download

    # Download from ggerganov/whisper.cpp repository
    model_path = hf_hub_download(
        repo_id="ggerganov/whisper.cpp",
        filename=name,
    )
    return model_path


def recombine_bpe_tokens_into_words(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recombine raw BPE subword tokens into intact full words with timestamps.

    In whisper.cpp BPE tokenization:
    - Tokens starting with a space (e.g. ' gener') represent the start of a new word.
    - Continuation subwords (e.g. 'ative'), contractions (e.g. \"'d\", \"'t\"), and
      punctuation (e.g. \",\", \".\") attach to the previous word.
    """
    words: List[Dict[str, Any]] = []

    contractions = {"'d", "'s", "'t", "'m", "'re", "'ve", "'ll", "n't"}
    trailing_punct = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "、", "。", "！", "？", "，"}

    for tok in tokens:
        raw_text = tok.get("text", "")
        # Skip special tokens e.g. [_BEG_], [_TT_123], [BLANK_AUDIO]
        if raw_text.startswith("[_") and raw_text.endswith("]"):
            continue

        clean_text = raw_text.strip()
        if not clean_text:
            continue

        t_offsets = tok.get("offsets", {})
        t_start = round(t_offsets.get("from", 0) / 1000.0, 3)
        t_end = round(t_offsets.get("to", 0) / 1000.0, 3)

        has_leading_space = raw_text.startswith(" ")
        starts_cjk = is_cjk_char(clean_text[0])
        is_contraction = clean_text.lower() in contractions or (clean_text.startswith("'") and len(clean_text) <= 3)
        is_punct = clean_text in trailing_punct

        if words and (is_contraction or is_punct or (not has_leading_space and not starts_cjk)):
            # Recombine continuation subword, contraction, or punctuation into previous word
            words[-1]["text"] += clean_text
            words[-1]["end"] = max(words[-1]["end"], t_end)
        else:
            words.append({
                "text": clean_text,
                "start": t_start,
                "end": t_end,
                "speaker": None,
            })

    return words


def normalize_whispercpp_output(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert whisper.cpp full JSON output to the common pipeline dict format."""
    detected_lang = raw_json.get("result", {}).get("language", "unknown")
    transcription = raw_json.get("transcription", [])

    segments: List[Dict[str, Any]] = []
    all_words: List[Dict[str, Any]] = []

    for seg in transcription:
        offsets = seg.get("offsets", {})
        start_sec = round(offsets.get("from", 0) / 1000.0, 3)
        end_sec = round(offsets.get("to", 0) / 1000.0, 3)
        seg_text = seg.get("text", "").strip()

        tokens = seg.get("tokens", [])
        words = recombine_bpe_tokens_into_words(tokens)
        all_words.extend(words)

        segments.append({
            "start": start_sec,
            "end": end_sec,
            "text": seg_text,
            "speaker": None,
            "words": words,
        })

    full_text = " ".join(s["text"] for s in segments if s["text"]).strip()

    return {
        "text": full_text,
        "language": detected_lang,
        "segments": segments,
        "words": all_words,
        "speaker_turns": [],
    }


def transcribe_whispercpp(
    audio_path: str,
    *,
    model_name: str = "large-v3",
    language: Optional[str] = None,
    threads: Optional[int] = None,
    no_gpu: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Dict[str, Any]:
    """Run whisper.cpp transcription using whisper-cli and return common result dict."""
    binary = find_whisper_cli()

    if progress and task_id is not None:
        progress.update(task_id, completed=1.0, description="[2/5] Loading whisper.cpp model...")

    model_file = resolve_whispercpp_model(model_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_prefix = str(Path(tmp_dir) / "whispercpp_out")

        cmd = [
            binary,
            "-m", model_file,
            "-f", audio_path,
            "--output-json-full",
            "-ojf",
            "-of", output_prefix,
            "-np",   # No prints to stdout
            "-fa",   # Flash attention
            "-sow",  # Split on word rather than on token
        ]

        if language:
            cmd.extend(["-l", language])
        else:
            cmd.extend(["-l", "auto"])

        if threads:
            cmd.extend(["-t", str(threads)])

        if no_gpu:
            cmd.append("-ng")

        if progress and task_id is not None:
            progress.update(task_id, description="[2/5] Transcribing speech (whisper.cpp Metal GPU)...")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli failed (code {proc.returncode}):\n{proc.stderr}")

        if progress and task_id is not None:
            progress.update(task_id, completed=3.0, description="[3/5] Processing timestamps...")

        json_file = Path(f"{output_prefix}.json")
        if not json_file.exists():
            raise FileNotFoundError(f"whisper-cli expected output file not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

    return normalize_whispercpp_output(raw_json)
