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

# 1. Create virtual environment and install all dependencies
echo ""
echo "[1/2] Installing dependencies and CLI tools..."
uv sync --extra dev

# Ensure macOS does not hide .pth files (Python 3.12+ skips UF_HIDDEN files)
if [[ "$OSTYPE" == "darwin"* ]]; then
    chflags -R nohidden .venv 2>/dev/null || true
fi

# 2. Download and cache model weights so they are permanently ready offline
echo ""
echo "[2/2] Checking and caching AI model weights (one-time setup)..."
if [ -n "$HF_TOKEN" ]; then
    uv run download-weights --hf-token "$HF_TOKEN"
else
    uv run download-weights
fi

echo ""
echo "=== Setup complete! ==="
echo "All model weights are cached locally. The pipeline will run without re-downloading."
echo ""
echo "To transcribe a recording:"
echo "  uv run transcribe \"path/to/recording.mp4\""
