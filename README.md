# Video-to-Text Transcription Pipeline

**Automated transcription of recorded interviews, lectures, and focus groups with word-level timestamps and speaker labels.**

This pipeline extracts audio from any video or audio file, transcribes it using Qwen3-ASR (supporting 52 languages), aligns word-level timestamps via forced alignment, and optionally identifies individual speakers through diarization. It produces transcripts in five formats: plain text, SRT, WebVTT, TSV (InqScribe-compatible), and structured JSON.

---

## Prerequisites

| Requirement            | Purpose                                | Installation |
|------------------------|----------------------------------------|-------------|
| **Python 3.11 or 3.12** | Runtime environment                   | Installed automatically by `uv` |
| **uv**                 | Python package and environment manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **FFmpeg**             | Audio extraction from media files      | Mac: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg` |

> [!IMPORTANT]
> **Windows**: Use PowerShell. Replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/misc-an/video-transcription-pipeline.git
cd video-transcription-pipeline
```

### 2. Set up once

Run the setup script:
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

This installs all dependencies and caches the AI model weights locally (~3 GB in `~/.cache/huggingface/hub/`). Once this one-time setup finishes, the pipeline runs locally without re-downloading weights.

*(Manual alternative using `uv`: `uv sync && uv run download-weights`)*

---

## Usage

### Basic transcription

```bash
uv run transcribe "path/to/recording.mp4" -l English
```

Output files are saved alongside the input file by default.

> [!TIP]
> **Performance Boost with `-l English` (Up to 4x Faster)**:
> Always pass the language flag (e.g., `-l English` or `-l en`). Bypassing the language auto-detection probing drops transcription time from **~5 minutes down to ~1 minute 15 seconds** for a 6-minute recording on Apple Silicon M4, while also eliminating accidental language switching on accents.

### Common examples

```bash
# Full interview with speaker labels and English language acceleration
uv run transcribe "interview.mp4" --diarize -l English

# Test with the first 30 seconds
uv run transcribe "interview.mp4" --duration 30

# Specify an output directory
uv run transcribe "interview.mp4" --output-dir "data/output" -l English
```

### Running from any directory

Install system-wide so `transcribe` works without `uv run` and from any path:

```bash
uv tool install .
transcribe "/path/to/any/recording.mp4"
```

Or point `uv` to the project when working from an external folder:
```bash
uv run --project "/path/to/video-transcription-pipeline" transcribe recording.mp4
```

---

## Speaker Diarization

Speaker diarization labels each segment with the speaker who said it (`SPEAKER_00`, `SPEAKER_01`, etc.). It requires a free [Hugging Face](https://huggingface.co) account with access granted to:

- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Run with speaker labels

```bash
uv run transcribe "interview.mp4" --diarize --hf-token "hf_your_token_here"
```

Or export the token once and diarization activates automatically:
```bash
export HF_TOKEN="hf_your_token_here"
uv run transcribe "interview.mp4"
```

Use `--no-diarize` to suppress automatic diarization when a token is set.

If the number of speakers is known:
```bash
uv run transcribe "interview.mp4" --diarize --min-speakers 2 --max-speakers 2
```

---

## Output Formats

For an input file `interview.mp4`, the following files are generated:

| File                      | Format     | Use Case |
|---------------------------|------------|----------|
| `interview.txt`           | Plain text | Reading, reviewing, qualitative coding |
| `interview.srt`           | SubRip     | Video players (VLC, YouTube) |
| `interview.vtt`           | WebVTT     | Web browsers, HTML5 video |
| `interview.tsv`           | Tab-delimited | [InqScribe](https://www.inqscribe.com/) import |
| `interview.json`          | JSON       | Programmatic analysis (NVivo, ATLAS.ti, Python, R) |
| `interview.extracted.wav` | WAV        | Extracted 16 kHz mono audio source |

### Sample output (.txt)

```
[00:00:02.870 - 00:00:04.410] SPEAKER_00: Good evening, ladies and gentlemen.

[00:00:05.390 - 00:00:26.950] SPEAKER_01: It is my privilege to introduce to you
Emeritus Professor of Physics, Emeritus Dean of Engineering...
```

> [!TIP]
> Pass `--cleanup-audio` to automatically delete the intermediate `.extracted.wav` file.

---

## Supported Media Formats

- **Video**: `.mp4`, `.webm`, `.mkv`, `.mov`, `.avi`, `.flv`, `.wmv`, `.m4v`, `.ts`
- **Audio**: `.mp3`, `.m4a`, `.aac`, `.flac`, `.wav`, `.ogg`, `.opus`

FFmpeg normalizes all inputs to 16 kHz 16-bit mono PCM WAV internally.

---

## Pipeline Architecture

```
Recording  →  FFmpeg (audio extraction)
           →  Qwen3-ASR 1.7B (speech-to-text, 52 languages)
           →  Qwen3 Forced Aligner 0.6B (word-level timestamps)
           →  PyAnnote Diarization (speaker labels, optional)
           →  Export (.txt, .srt, .vtt, .tsv, .json)
```

Hardware acceleration is automatic: NVIDIA GPU (CUDA), Apple Silicon (Metal/MPS), or CPU fallback.

### Running Speed & Performance
- **Speed**: Transcribes at **~4x to 5x faster than real-time** when the language flag is specified (e.g. a 6-minute interview transcribes in **~1m 15s** on Apple Silicon M4).
- **Clean Single-Line Progress**: A single unified progress bar tracks all 5 pipeline stages and chunk progress in real time with an active countdown timer (ETA).

---

## CLI Reference

| Option | Description | Default |
|--------|------------|---------|
| `media_file` | Path to video or audio file | *(required)* |
| `-o`, `--output-dir` | Output directory | Input file directory |
| `--model` | ASR model ID | `Qwen/Qwen3-ASR-1.7B` |
| `--aligner-model` | Forced aligner model ID | `Qwen/Qwen3-ForcedAligner-0.6B` |
| `-l`, `--language` | Recording language (e.g. `English`, `en`) | Auto-detect |
| `--device` | Compute device (`cpu`, `cuda`, `mps`) | Auto-detect |
| `--no-align` | Skip word-level alignment | Alignment enabled |
| `--diarize` | Enable speaker diarization | Off (auto-on with token) |
| `--no-diarize` | Disable speaker diarization | — |
| `--hf-token` | Hugging Face access token | `$HF_TOKEN` env var |
| `--min-speakers` | Minimum expected speakers | Automatic |
| `--max-speakers` | Maximum expected speakers | Automatic |
| `--start-time` | Start offset (e.g. `00:05:00`) | `0` |
| `--duration` | Process only N seconds | Full file |
| `--cleanup-audio` | Delete intermediate WAV after export | Keep |

---

## Project Structure

```
video-transcription-pipeline/
├── README.md
├── pyproject.toml                 # Dependencies and build config
├── requirements.txt               # Locked dependency versions
├── uv.lock                        # Deterministic lockfile
│
├── scripts/
│   ├── setup_env.sh               # Environment bootstrap
│   ├── transcribe_pipeline.py     # Standalone pipeline runner
│   └── download_weights.py        # Standalone model downloader
│
├── src/video_transcription/
│   ├── __init__.py
│   ├── pipeline.py                # Core transcription and diarization logic
│   ├── audio.py                   # FFmpeg audio extraction
│   ├── export.py                  # Output formatters (TXT, SRT, VTT, TSV, JSON)
│   └── weights.py                 # Model download utilities
│
├── tests/
│   └── test_pipeline.py           # Unit tests
│
├── data/
│   ├── input/                     # Place recordings here (optional)
│   └── output/                    # Default output location (optional)
│
└── examples/                      # Sample outputs from a verified run
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FFmpeg binary not found` | Install FFmpeg: Mac `brew install ffmpeg`, Linux `sudo apt install ffmpeg` |
| `ModuleNotFoundError: No module named 'video_transcription'` | Run `uv sync` from inside the project directory |
| Out of memory or very slow | Try `--duration 60` for a shorter segment, or `--device cpu` to reduce memory usage |
| Speaker diarization fails | Verify: (1) valid HF token, (2) accepted both model agreements, (3) `--diarize` flag or `HF_TOKEN` exported |

---

## Running Tests

```bash
uv run --extra dev pytest
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **Transcription** | Converting spoken words in a recording into written text |
| **Diarization** | Identifying which speaker said what and when |
| **ASR** | Automatic Speech Recognition — AI-based speech-to-text |
| **Forced alignment** | Mapping each recognized word to its precise time in the audio |
| **FFmpeg** | Open-source tool for reading and converting audio/video files |
| **SRT / VTT** | Standard subtitle formats for video players and web |
| **TSV** | Tab-Separated Values — a text-based tabular format |

---

## License

This project is open source. See the repository for license details.
