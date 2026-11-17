# Video-to-Text Transcription Pipeline

This tool converts spoken words in audio and video files into text transcripts. It adds word-level timestamps and speaker labels.

The tool supports two speech recognition engines:
- **Qwen3-ASR**: Runs on Apple Silicon GPU (MPS), NVIDIA GPU (CUDA), and CPU.
- **whisper.cpp**: Runs on Apple Silicon GPU (Metal) and CPU.

The tool writes transcripts in five formats: plain text (.txt), SubRip (.srt), WebVTT (.vtt), InqScribe tab-delimited (.tsv), and JSON (.json).

---

## Prerequisites

Install these tools before you start:

| Tool | Purpose | Installation Command |
|---|---|---|
| **Python 3.11 or 3.12** | Runtime environment | Installed automatically by `uv` |
| **uv** | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **FFmpeg** | Audio extraction | macOS: `brew install ffmpeg`<br>Linux: `sudo apt install ffmpeg` |
| **whisper.cpp** *(optional)* | Metal GPU speech engine | macOS: `brew install whisper-cpp` |

> [!IMPORTANT]
> **Windows**: Use PowerShell. Replace `source .venv/bin/activate` with `.venv\Scripts\Activate.ps1`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/misc-an/video-transcription-pipeline.git
cd video-transcription-pipeline
```

### 2. Run the setup script

Make the script executable and run it:

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Select one option when prompted:

1. **Option 1**: Install Qwen3-ASR only (~3 GB download).
2. **Option 2**: Install whisper.cpp only (~1.6 GB download).
3. **Option 3**: Install both engines (~4.6 GB download).

The script installs all dependencies and downloads the model files. The files are saved in `~/.cache/huggingface/hub/`.

---

## Usage

### 1. Basic Transcription

Run this command to transcribe a file:

```bash
uv run transcribe "recording.mp4" -l English
```

The tool writes output files to `data/output/` by default.

### 2. Select the Speech Engine

Set the engine with the `--engine` option:

```bash
# Use Qwen3-ASR (default)
uv run transcribe "recording.mp4" --engine qwen -l English

# Use whisper.cpp (Apple Silicon Metal GPU)
uv run transcribe "recording.mp4" --engine whispercpp -l English
```

To use a specific whisper.cpp model, use `--whisper-model`:

```bash
# Use 5-bit quantized model (uses ~1.6 GB RAM)
uv run transcribe "recording.mp4" --engine whispercpp --whisper-model large-v3-q5_0
```

### 3. Add Speaker Labels

Speaker diarization identifies different speakers (`SPEAKER_00`, `SPEAKER_01`).

To use speaker diarization, complete these steps:
1. Create a free account on [Hugging Face](https://huggingface.co).
2. Accept the user conditions for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Accept the user conditions for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
4. Create an access token on Hugging Face.

Set the token in your environment:

```bash
export HF_TOKEN="hf_your_token_here"
```

Run transcription with the `--diarize` flag:

```bash
uv run transcribe "interview.mp4" --diarize -l English
```

If you know the speaker count, set the limits:

```bash
uv run transcribe "interview.mp4" --diarize --min-speakers 2 --max-speakers 2
```

### 4. Common Command Examples

```bash
# Process only the first 30 seconds
uv run transcribe "interview.mp4" --duration 30

# Start at 1 minute and process for 60 seconds
uv run transcribe "interview.mp4" --start-time 00:01:00 --duration 60

# Set a custom output folder
uv run transcribe "interview.mp4" -o "my_transcripts"

# Include timestamps in the plain text output
uv run transcribe "interview.mp4" --timestamps-in-txt
```

---

## Output Files

For an input file `interview.mp4`, the tool creates these files:

| File | Format | Purpose |
|---|---|---|
| `interview.txt` | Plain text | Read, review, or edit text |
| `interview.srt` | SubRip | Subtitles for video players (VLC, YouTube) |
| `interview.vtt` | WebVTT | Subtitles for web browsers |
| `interview.tsv` | Tab-delimited | Import into InqScribe |
| `interview.json` | JSON | Read programmatically in Python or R |
| `interview.extracted.wav` | WAV | Extracted 16 kHz mono audio file |

---

## Supported Media Formats

The tool accepts these file extensions:

- **Video**: `.mp4`, `.webm`, `.mkv`, `.mov`, `.avi`, `.flv`, `.wmv`, `.m4v`, `.ts`
- **Audio**: `.mp3`, `.m4a`, `.aac`, `.flac`, `.wav`, `.ogg`, `.opus`

FFmpeg converts all inputs to 16 kHz 16-bit mono WAV before processing.

---

## Architecture

The pipeline executes five steps:

```
1. Audio Extraction (FFmpeg)
   ↓
2. Speech Recognition (Qwen3-ASR or whisper.cpp)
   ↓
3. Word Alignment (Forced Aligner or Attention DTW)
   ↓
4. Speaker Diarization (PyAnnote, optional)
   ↓
5. File Export (.txt, .srt, .vtt, .tsv, .json)
```

Hardware acceleration activates automatically:
- Apple Silicon: Metal GPU for whisper.cpp, MPS for Qwen3-ASR and PyAnnote.
- NVIDIA: CUDA for Qwen3-ASR and PyAnnote.
- Other: CPU fallback.

---

## Command-Line Options

| Option | Description | Default |
|---|---|---|
| `media_file` | Path to audio or video file | *(Required)* |
| `-o`, `--output-dir` | Output folder path | `data/output/` |
| `--engine` | Engine choice: `qwen` or `whispercpp` | Auto-detect (prefers `qwen`) |
| **Qwen Options** | | |
| `--model` | Hugging Face model ID | `Qwen/Qwen3-ASR-1.7B` |
| `--aligner-model` | Forced aligner model ID | `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--chunk-size` | Chunk length in seconds | `30.0` |
| **whisper.cpp Options** | | |
| `--whisper-model` | GGML model name or path | `large-v3` |
| `--whisper-threads` | Number of compute threads | System default |
| `--no-gpu` | Disable Metal GPU and use CPU | `false` |
| **Shared Options** | | |
| `-l`, `--language` | Language code (for example: `en`, `English`) | Auto-detect |
| `--device` | Device name: `cpu`, `cuda`, or `mps` | Auto-detect |
| `--no-align` | Disable word alignment (Qwen only) | `false` |
| `--diarize` | Enable speaker diarization | `false` |
| `--no-diarize` | Disable speaker diarization | `false` |
| `--hf-token` | Hugging Face token | `$HF_TOKEN` |
| `--min-speakers` | Minimum number of speakers | Auto-detect |
| `--max-speakers` | Maximum number of speakers | Auto-detect |
| `--start-time` | Start offset (seconds or `HH:MM:SS`) | `0` |
| `--duration` | Process duration in seconds | Full file |
| `--timestamps-in-txt` | Add timestamps to the .txt file | `false` |
| `--cleanup-audio` | Delete intermediate WAV file | `false` |

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `FFmpeg binary not found` | FFmpeg is not installed or not in PATH | Install FFmpeg: `brew install ffmpeg` |
| `whisper-cli not found` | whisper.cpp is not installed | Install whisper.cpp: `brew install whisper-cpp` |
| `ASR engine 'qwen' is not available` | Qwen dependencies are not installed | Run: `uv sync --extra qwen` |
| Out of memory error | Model is too large for system RAM | Use: `--engine whispercpp --whisper-model large-v3-q5_0` |
| Diarization fails | Missing token or model agreement | Export valid `HF_TOKEN` and accept terms on Hugging Face |

---

## Verification Tests

Run the test suite with this command:

```bash
uv run --extra dev pytest
```

---

## License

This project is licensed under the MIT License.
