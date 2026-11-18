#!/usr/bin/env python3
"""Bulk voice-clone narrations with adjustable rate and punctuation pauses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from bulk_generate import (
    append_progress,
    choose_device,
    estimate_token_budget,
    gpu_summary,
    load_model,
    read_narrations,
    save_audio,
    slug,
)


ROOT = Path(__file__).resolve().parent
CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PIECE_PATTERN = re.compile(r"[^,;.!?]+[,;.!?]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "narrations.txt")
    parser.add_argument("--reference-audio", type=Path, default=ROOT / "clone.mp3")
    parser.add_argument("--output", type=Path, default=ROOT / "output-voice-clone")
    parser.add_argument("--model", default=CLONE_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, or cpu")
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N narrations; 0 means all")
    parser.add_argument("--speed", type=float, default=1.0, help="Pitch-preserving speech rate; 1.0 is native, below 1 is slower")
    parser.add_argument("--sentence-pause", type=float, default=0.6, help="Exact silence between sentences, in seconds")
    parser.add_argument("--comma-pause", type=float, default=0.0, help="Exact silence after commas and semicolons, in seconds")
    parser.add_argument("--ending-silence", type=float, default=1.0)
    parser.add_argument("--ending-fade-ms", type=float, default=100.0)
    parser.add_argument("--max-token-cap", type=int, default=8192)
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already exist")
    return parser.parse_args()


def split_punctuation_pieces(text: str) -> list[tuple[str, float]]:
    """Return text chunks paired with the pause after their punctuation."""
    matches = list(PIECE_PATTERN.finditer(text))
    pieces: list[tuple[str, float]] = []
    for index, match in enumerate(matches):
        piece = match.group().strip()
        if not piece:
            continue
        ending = piece[-1]
        has_following_text = any(m.group().strip() for m in matches[index + 1 :])
        if has_following_text and ending in ".!?":
            pause = -1.0  # replaced by --sentence-pause in main
        elif has_following_text and ending in ",;":
            pause = -2.0  # replaced by --comma-pause in main
        else:
            pause = 0.0
        pieces.append((piece, pause))
    return pieces


def as_mono_float32(wav: np.ndarray) -> np.ndarray:
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)
    return np.nan_to_num(audio, copy=False)


def stretch_speech(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    import librosa

    # librosa's phase-vocoder stretch changes duration while preserving pitch.
    return np.asarray(librosa.effects.time_stretch(audio, rate=speed), dtype=np.float32)


def main() -> int:
    args = parse_args()
    if not 0.5 <= args.speed <= 1.25:
        print("--speed must be between 0.5 and 1.25 (1.0 is native speed).", file=sys.stderr)
        return 2
    if args.sentence_pause < 0 or args.comma_pause < 0 or args.ending_silence < 0:
        print("Pause and ending-silence values cannot be negative.", file=sys.stderr)
        return 2
    if args.limit < 0 or args.max_token_cap < 512:
        print("--limit cannot be negative and --max-token-cap must be at least 512.", file=sys.stderr)
        return 2

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    ref_path = args.reference_audio if args.reference_audio.is_absolute() else ROOT / args.reference_audio
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    if not input_path.is_file():
        print(f"Input text file not found: {input_path}", file=sys.stderr)
        return 2
    if not ref_path.is_file():
        print(f"Voice reference not found: {ref_path}", file=sys.stderr)
        return 2

    texts = read_narrations(input_path)
    if args.limit:
        texts = texts[: args.limit]
    if not texts:
        print("No narration paragraphs found.", file=sys.stderr)
        return 2

    try:
        import soundfile  # noqa: F401
        import torch
    except ImportError as exc:
        print(f"Missing Python dependency: {exc}. Run setup_and_run.sh.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        device, device_index = choose_device(torch, args.device)
        model = load_model(torch, args.model, device)
    except Exception as exc:
        print(f"Could not initialize Qwen on the requested device: {exc}", file=sys.stderr)
        return 2

    print(f"Creating one reusable speaker prompt from {ref_path.name}.", flush=True)
    try:
        # Preserve the same x-vector cloning mode used for the accepted samples.
        prompt = model.create_voice_clone_prompt(ref_audio=str(ref_path), x_vector_only_mode=True)
    except Exception as exc:
        print(f"Could not create the voice prompt: {exc}", file=sys.stderr)
        return 2

    ref_hash = hashlib.sha256(ref_path.read_bytes()).hexdigest()[:8]
    settings = {
        "model": args.model,
        "reference": ref_hash,
        "speed": args.speed,
        "sentence_pause": args.sentence_pause,
        "comma_pause": args.comma_pause,
        "ending_silence": args.ending_silence,
        "ending_fade_ms": args.ending_fade_ms,
        "max_token_cap": args.max_token_cap,
    }
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    progress_path = output_dir / "progress.jsonl"
    run_id = str(time.time_ns())
    pauses = { -1.0: args.sentence_pause, -2.0: args.comma_pause }
    print(
        f"Narrations: {len(texts)} | speed={args.speed:.2f} | "
        f"sentence pause={args.sentence_pause:.2f}s | comma pause={args.comma_pause:.2f}s | device={device}",
        flush=True,
    )

    completed = 0
    for number, text in enumerate(texts, 1):
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        out_path = output_dir / f"{number:04d}_{slug(text)}_{text_hash}_{settings_hash}.wav"
        if out_path.exists() and not args.force:
            try:
                import soundfile as sf
                if sf.info(out_path).frames > 0:
                    print(f"SKIP {number}/{len(texts)}: {out_path.name}", flush=True)
                    continue
            except Exception:
                pass

        chunks = split_punctuation_pieces(text)
        audio_parts: list[np.ndarray] = []
        sample_rate: int | None = None
        for piece_index, (piece, pause_marker) in enumerate(chunks, 1):
            token_cap = estimate_token_budget(piece, args.max_token_cap)
            print(
                f"Generating {number}/{len(texts)}, phrase {piece_index}/{len(chunks)} "
                f"(max_new_tokens={token_cap})...",
                flush=True,
            )
            try:
                wavs, sample_rate = model.generate_voice_clone(
                    text=piece,
                    language="English",
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    max_new_tokens=token_cap,
                )
            except Exception as exc:
                if device_index is not None and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                print(f"FAILED narration {number}, phrase {piece_index}: {exc}", file=sys.stderr, flush=True)
                return 1

            speech = stretch_speech(as_mono_float32(wavs[0]), args.speed)
            audio_parts.append(speech)
            pause_s = pauses.get(pause_marker, 0.0)
            if pause_s > 0 and piece_index < len(chunks):
                audio_parts.append(np.zeros(round(sample_rate * pause_s), dtype=np.float32))

        if sample_rate is None or not audio_parts:
            print(f"No audio returned for narration {number}.", file=sys.stderr, flush=True)
            return 1
        composite = np.concatenate(audio_parts)
        duration = save_audio(out_path, composite, sample_rate, args.ending_fade_ms, args.ending_silence)
        completed += 1
        print(f"SAVED {number}/{len(texts)}: {out_path} ({duration:.1f}s)", flush=True)
        append_progress(progress_path, {
            "run_id": run_id,
            "item": number,
            "status": "saved",
            "output": str(out_path),
            "duration_s": round(duration, 3),
            "sample_rate": int(sample_rate),
            "speed": args.speed,
            "sentence_pause_s": args.sentence_pause,
            "comma_pause_s": args.comma_pause,
            "device": device,
        })

    print(f"COMPLETE: {completed}/{len(texts)} saved in {output_dir}", flush=True)
    if device_index is not None:
        free_gb, total_gb = gpu_summary(torch, device_index)
        print(f"GPU memory after generation: {free_gb:.1f}/{total_gb:.1f} GiB free", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
