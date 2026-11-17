#!/usr/bin/env bash
set -e

# Always run from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Setting up Video-to-Text Transcription Pipeline ==="
echo "Project root: $ROOT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg is not found in your system path."
    if [ -f "/opt/homebrew/bin/ffmpeg" ]; then
        echo "Found ffmpeg at /opt/homebrew/bin/ffmpeg — adding to PATH."
        export PATH="/opt/homebrew/bin:$PATH"
    else
        echo "Please install ffmpeg before running transcriptions."
        echo "  Mac:   brew install ffmpeg"
        echo "  Linux: sudo apt install ffmpeg"
    fi
else
    echo "Found ffmpeg at: $(command -v ffmpeg)"
fi

# Check if whisper-cli is installed
if command -v whisper-cli &> /dev/null; then
    echo "Found whisper.cpp at: $(command -v whisper-cli)"
elif [ -f "/opt/homebrew/bin/whisper-cli" ]; then
    echo "Found whisper.cpp at /opt/homebrew/bin/whisper-cli — adding to PATH."
    export PATH="/opt/homebrew/bin:$PATH"
fi

# Engine selection menu
if [ -n "$1" ]; then
    engine_choice="$1"
elif [ -n "$ENGINE_CHOICE" ]; then
    engine_choice="$ENGINE_CHOICE"
else
    echo ""
    echo "Select which ASR engine(s) to set up:"
    echo ""
    echo "  1) Qwen3-ASR only       (~3 GB, native PyTorch / Apple MPS / CUDA)"
    echo "  2) whisper.cpp only     (Native Apple Silicon Metal GPU / C++)"
    echo "  3) Both                 (Both Qwen3 and whisper.cpp)"
    echo ""
    read -rp "Enter choice [1/2/3] (default: 1): " engine_choice
fi

engine_choice="${engine_choice:-1}"

case "$engine_choice" in
    1|qwen)
        echo "→ Setting up Qwen3-ASR backend..."
        EXTRAS="qwen"
        engine_arg="1"
        ;;
    2|whisper|whispercpp)
        echo "→ Setting up whisper.cpp backend..."
        EXTRAS=""
        engine_arg="2"
        ;;
    3|both|all)
        echo "→ Setting up both backends (Qwen3-ASR and whisper.cpp)..."
        EXTRAS="all"
        engine_arg="3"
        ;;
    *)
        echo "Invalid choice '$engine_choice'. Defaulting to Qwen3-ASR."
        EXTRAS="qwen"
        engine_arg="1"
        ;;
esac

# Ensure whisper-cli is installed if chosen
if [ "$engine_arg" = "2" ] || [ "$engine_arg" = "3" ]; then
    if ! command -v whisper-cli &> /dev/null && [ ! -f "/opt/homebrew/bin/whisper-cli" ]; then
        echo ""
        echo "Installing whisper.cpp via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install whisper-cpp
        else
            echo "Warning: Homebrew not found. Please install whisper-cpp manually: brew install whisper-cpp"
        fi
    fi
fi

# 1. Create virtual environment and install all dependencies
echo ""
echo "[1/2] Installing dependencies and CLI tools..."
if [ -n "$EXTRAS" ]; then
    uv sync --extra "$EXTRAS" --extra dev
else
    uv sync --extra dev
fi

# Ensure macOS does not hide .pth files (Python 3.12+ skips UF_HIDDEN files)
if [[ "$OSTYPE" == "darwin"* ]]; then
    chflags -R nohidden .venv 2>/dev/null || true
fi

# 2. Download and cache model weights so they are permanently ready offline
echo ""
echo "[2/2] Checking and caching AI model weights (one-time setup)..."
if [ -n "$HF_TOKEN" ]; then
    uv run download-weights --engine "$engine_arg" --hf-token "$HF_TOKEN"
else
    uv run download-weights --engine "$engine_arg"
fi

echo ""
echo "=== Setup complete! ==="
echo "All model weights are cached locally. The pipeline will run without re-downloading."
echo ""
echo "To transcribe a recording:"
if [ "$engine_arg" = "2" ]; then
    echo "  uv run transcribe \"path/to/recording.mp4\" --engine whispercpp -l English"
elif [ "$engine_arg" = "3" ]; then
    echo "  uv run transcribe \"path/to/recording.mp4\"                    # Uses default Qwen3"
    echo "  uv run transcribe \"path/to/recording.mp4\" --engine whispercpp # Uses whisper.cpp (Metal GPU)"
else
    echo "  uv run transcribe \"path/to/recording.mp4\" -l English"
fi
