"""Qwen3-ASR backend: transcription and forced alignment using Qwen3-ASR 1.7B."""

from typing import Any, Dict, List, Optional

import torch
from rich.progress import Progress, TaskID


def transcribe_qwen(
    audio_path: str,
    *,
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    language: Optional[str] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    do_align: bool = True,
    chunk_size: float = 30.0,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Dict[str, Any]:
    """Run Qwen3-ASR transcription and return the common pipeline result dict.

    Returns:
        dict with keys: text, language, segments, words, speaker_turns
    """
    from ..pipeline import group_words_into_segments

    import qwen_asr.inference.qwen3_asr as qwen_asr_mod
    import qwen_asr.inference.utils as qwen_utils
    from qwen_asr import Qwen3ASRModel

    # Set optimal chunk duration to prevent autoregressive hallucination loops on long audio
    chunk_sec = float(chunk_size) if chunk_size else 30.0
    qwen_asr_mod.MAX_FORCE_ALIGN_INPUT_SECONDS = chunk_sec
    qwen_utils.MAX_FORCE_ALIGN_INPUT_SECONDS = chunk_sec
    qwen_asr_mod.MAX_ASR_INPUT_SECONDS = chunk_sec
    qwen_utils.MAX_ASR_INPUT_SECONDS = chunk_sec

    # Standardize language code/name for Qwen3 if provided (e.g. 'en' -> 'English')
    lang_map = {
        "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
        "de": "German", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
        "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi",
    }
    qwen_lang = lang_map.get(language.lower(), language) if language else None

    aligner_kwargs = dict(device_map=device, dtype=dtype) if do_align else None

    # Load from local cache first (fast, offline, no repeated network calls); fall back to downloading
    if progress and task_id is not None:
        progress.update(task_id, completed=1.0, description="[2/5] Loading Qwen3-ASR model...")

    model = None
    try:
        local_aligner_kwargs = dict(aligner_kwargs, local_files_only=True) if aligner_kwargs else None
        model = Qwen3ASRModel.from_pretrained(
            asr_model_name,
            dtype=dtype,
            device_map=device,
            local_files_only=True,
            forced_aligner=aligner_model_name if do_align else None,
            forced_aligner_kwargs=local_aligner_kwargs,
        )
    except Exception:
        model = Qwen3ASRModel.from_pretrained(
            asr_model_name,
            dtype=dtype,
            device_map=device,
            forced_aligner=aligner_model_name if do_align else None,
            forced_aligner_kwargs=aligner_kwargs,
        )

    # Enforce sequential chunk alignment (batch size 1) to avoid MPS tensor padding zero-timestamp bug
    model.max_inference_batch_size = 1

    # Attach single overall progress tracker to chunk-level ASR inference
    curr_chunk = 0
    total_asr_chunks = 1
    if progress and task_id is not None and hasattr(model, "_infer_asr_transformers"):
        def infer_asr_with_progress(contexts, wavs, languages):
            nonlocal curr_chunk, total_asr_chunks
            total_chunks = len(wavs)
            total_asr_chunks = total_chunks
            batch_size = 1 if device == "mps" else (model.max_inference_batch_size if model.max_inference_batch_size > 0 else 1)

            outs = []
            texts = [model._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

            for i in range(0, len(texts), batch_size):
                sub_text = texts[i : i + batch_size]
                sub_wavs = wavs[i : i + batch_size]
                curr_chunk += len(sub_text)
                progress.update(
                    task_id,
                    completed=1.0 + (curr_chunk / total_chunks),
                    description=f"[2/5] Transcribing speech (chunk {curr_chunk}/{total_chunks})...",
                )

                inputs = model.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
                inputs = inputs.to(model.model.device).to(model.model.dtype)

                text_ids = model.model.generate(
                    **inputs,
                    max_new_tokens=model.max_new_tokens,
                    repetition_penalty=1.1,
                )

                decoded = model.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                outs.extend(list(decoded))

            return outs

        model._infer_asr_transformers = infer_asr_with_progress

    # Wrap forced aligner to track alignment progress per chunk
    align_curr = 0
    if do_align and progress and task_id is not None and hasattr(model, "forced_aligner") and model.forced_aligner is not None:
        orig_align = model.forced_aligner.align

        def align_with_progress(audio, text, language):
            nonlocal align_curr
            batch_count = len(audio) if isinstance(audio, list) else 1
            align_curr += batch_count
            denom = max(1, total_asr_chunks)
            progress.update(
                task_id,
                completed=2.0 + min(1.0, align_curr / denom),
                description=f"[3/5] Aligning timestamps (chunk {align_curr}/{denom})...",
            )
            return orig_align(audio=audio, text=text, language=language)

        model.forced_aligner.align = align_with_progress

    if progress and task_id is not None:
        progress.update(task_id, description="[2/5] Transcribing speech...")

    with torch.inference_mode():
        transcriptions = model.transcribe(
            audio=audio_path,
            language=qwen_lang,
            return_time_stamps=do_align,
        )

    res = transcriptions[0]
    detected_lang = res.language or (language or "unknown")
    full_text = res.text or ""

    if progress and task_id is not None:
        progress.update(task_id, completed=2.0)

    # Word Alignment Extraction
    if progress and task_id is not None:
        progress.update(task_id, completed=3.0, description="[3/5] Processing timestamps...")

    words: List[Dict[str, Any]] = []
    if do_align and res.time_stamps and hasattr(res.time_stamps, "items"):
        for item in res.time_stamps.items:
            w_start = float(item.start_time)
            w_end = float(item.end_time)
            if w_end <= w_start:
                w_end = round(w_start + 0.05, 3)
            words.append({
                "text": item.text,
                "start": w_start,
                "end": w_end,
                "speaker": None,
            })

    # Sanitize timestamps: ensure monotonically non-decreasing start times and valid positive durations
    for i in range(1, len(words)):
        if words[i]["start"] < words[i - 1]["start"]:
            words[i]["start"] = words[i - 1]["end"]
        if words[i]["end"] <= words[i]["start"]:
            words[i]["end"] = round(words[i]["start"] + 0.05, 3)

    # Build structured output segments
    if words:
        segments = group_words_into_segments(words)
    else:
        segments = [{
            "start": 0.0,
            "end": 0.0,
            "text": full_text,
            "speaker": None,
            "words": [],
        }]

    return {
        "text": full_text,
        "language": detected_lang,
        "segments": segments,
        "words": words,
        "speaker_turns": [],
    }
