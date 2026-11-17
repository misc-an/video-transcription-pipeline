"""ASR backend discovery and loading."""

from typing import List


def get_available_backends() -> List[str]:
    """Return list of installed backend names."""
    available = []
    try:
        import qwen_asr  # noqa: F401
        available.append("qwen")
    except ImportError:
        pass

    try:
        from .whispercpp_backend import find_whisper_cli
        find_whisper_cli()
        available.append("whispercpp")
    except (ImportError, FileNotFoundError):
        pass

    return available


def get_default_backend() -> str:
    """Return the best available backend, preferring Qwen3."""
    available = get_available_backends()
    if "qwen" in available:
        return "qwen"
    if "whispercpp" in available:
        return "whispercpp"
    raise ImportError(
        "No ASR backend available. Run setup_env.sh or install with:\n"
        "  uv sync --extra qwen     (for Qwen3-ASR)\n"
        "  brew install whisper-cpp (for whisper.cpp)\n"
        "  ./scripts/setup_env.sh   (interactive installer)"
    )
