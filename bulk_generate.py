#!/usr/bin/env python3
"""Resumable, resource-aware bulk narration generation with Qwen3-TTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CODEC_TOKENS_PER_SECOND = 12


@dataclass
class Narration:
    number: int
    text: str
    token_budget: int
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "narrations.txt")
    parser.add_argument("--voice", type=Path, default=ROOT / "voice_description.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, or cpu")
    parser.add_argument("--batch-size", default="auto", help="auto or a positive integer")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-token-cap", type=int, default=8192)
    parser.add_argument("--ending-silence", type=float, default=1.0)
    parser.add_argument("--ending-fade-ms", type=float, default=100.0)
    parser.add_argument("--force", action="store_true", help="Regenerate WAVs that already exist")
    return parser.parse_args()


def read_narrations(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8-sig")
    # Ignore comment-only template lines; blank paragraphs separate entries.
    lines = [line for line in raw.splitlines() if not line.lstrip().startswith("#")]
    blocks = re.split(r"\n\s*\n+", "\n".join(lines))
    result = [re.sub(r"\s+", " ", block).strip() for block in blocks]
    return [ensure_sentence_ending(text) for text in result if text]


def ensure_sentence_ending(text: str) -> str:
    # Give Qwen explicit prosodic closure so the final word is spoken as a
    # complete thought rather than as an unfinished fragment.
    if not re.search(r"[.!?…][\"'’”)]*$", text):
        return text + "."
    return text


def slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")[:42]
    return value or "narration"


def estimate_token_budget(text: str, hard_cap: int) -> int:
    words = max(1, len(text.split()))
    # 18 codec tokens/word allows a very slow read plus generous room for
    # pauses. The minimum prevents short clips from being cut at the tail.
    estimated = max(512, words * 18 + 384)
    return min(hard_cap, estimated)


def gpu_summary(torch: Any, device_index: int) -> tuple[float, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return free_bytes / (1024**3), total_bytes / (1024**3)


def choose_device(torch: Any, requested: str) -> tuple[str, int | None]:
    if requested == "cpu":
        return "cpu", None
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but PyTorch cannot see an NVIDIA/ROCm GPU.")
        index = int(requested.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {requested}, but only {torch.cuda.device_count()} GPU(s) are visible.")
        return requested, index
    if requested != "auto":
        raise ValueError("--device must be auto, cpu, or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA/ROCm GPU is visible to PyTorch. On a rented GPU, rerun setup_and_run.sh; "
            "if you intentionally want CPU inference, pass --device cpu."
        )
    best_index = 0
    best_free = -1.0
    for index in range(torch.cuda.device_count()):
        free_gb, _ = gpu_summary(torch, index)
        if free_gb > best_free:
            best_index, best_free = index, free_gb
    return f"cuda:{best_index}", best_index


def auto_batch_size(torch: Any, device_index: int | None, max_tokens: int, limit: int) -> int:
    if device_index is None:
        return 1
    free_gb, total_gb = gpu_summary(torch, device_index)
    # Keep a 2.5 GiB safety margin and budget more per concurrent narration
    # when the longest item has a larger token allowance.
    reserve_gb = max(2.5, total_gb * 0.08)
    per_item_gb = 1.4 + 1.25 * (max_tokens / 4096)
    usable_gb = max(0.0, free_gb - reserve_gb)
    size = max(1, int(usable_gb // per_item_gb))
    return min(size, limit)


def is_oom_error(exc: BaseException, torch: Any) -> bool:
    oom_class = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom_class) and isinstance(exc, oom_class)) or "out of memory" in str(exc).lower()


def append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def postprocess_audio(wav: Any, sample_rate: int, fade_ms: float, silence_s: float):
    import numpy as np

    audio = np.array(wav, dtype=np.float32, copy=True)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)
    audio = np.nan_to_num(audio, copy=False)
    fade_samples = min(len(audio), max(0, int(sample_rate * fade_ms / 1000)))
    if fade_samples > 1:
        audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    silence = np.zeros(max(0, int(sample_rate * silence_s)), dtype=np.float32)
    return np.concatenate((audio, silence))


def save_audio(path: Path, wav: Any, sample_rate: int, fade_ms: float, silence_s: float) -> float:
    import soundfile as sf

    audio = postprocess_audio(wav, sample_rate, fade_ms, silence_s)
    temporary = path.with_name(path.stem + ".partial.wav")
    sf.write(temporary, audio, sample_rate, subtype="PCM_24")
    temporary.replace(path)
    return len(audio) / sample_rate


def load_model(torch: Any, model_id: str, device: str):
    from qwen_tts import Qwen3TTSModel

    if device == "cpu":
        dtype = torch.float32
        print("WARNING: CPU mode is much slower than a rented GPU.", flush=True)
        kwargs: dict[str, Any] = {}
    else:
        index = int(device.split(":")[1])
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        name = torch.cuda.get_device_name(index)
        props = torch.cuda.get_device_properties(index)
        free_gb, total_gb = gpu_summary(torch, index)
        print(
            f"GPU {index}: {name} | {props.total_memory / 1024**3:.1f} GiB total | "
            f"{free_gb:.1f}/{total_gb:.1f} GiB free before model load | dtype={dtype}",
            flush=True,
        )
        if free_gb < 8.0:
            print(
                "WARNING: under 8 GiB is free. The 1.7B voice-design model may not fit; "
                "close other GPU jobs or rent a larger-memory GPU.",
                file=sys.stderr,
                flush=True,
            )
        kwargs = {"attn_implementation": "sdpa"}
    print(f"Loading {model_id} on {device} (one model copy)...", flush=True)
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
    )


def main() -> int:
    args = parse_args()
    cache_home = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface")).expanduser()
    if not cache_home.is_absolute():
        cache_home = ROOT / cache_home
    cache_home = cache_home.resolve()
    os.environ["HF_HOME"] = str(cache_home)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    voice_path = args.voice if args.voice.is_absolute() else ROOT / args.voice
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    if not input_path.is_file():
        print(f"Input text file not found: {input_path}", file=sys.stderr)
        return 2
    if not voice_path.is_file():
        print(f"Voice description not found: {voice_path}", file=sys.stderr)
        return 2
    texts = read_narrations(input_path)
    if not texts:
        print(f"No narration paragraphs found in {input_path}. Add one narration per paragraph.", file=sys.stderr)
        return 2
    if args.max_token_cap < 512:
        print("--max-token-cap must be at least 512.", file=sys.stderr)
        return 2
    if args.max_batch_size < 1:
        print("--max-batch-size must be positive.", file=sys.stderr)
        return 2
    if args.batch_size == "auto":
        requested_batch_size = None
    else:
        try:
            requested_batch_size = int(args.batch_size)
        except ValueError:
            print("--batch-size must be 'auto' or an integer.", file=sys.stderr)
            return 2
        if requested_batch_size < 1:
            print("--batch-size must be positive.", file=sys.stderr)
            return 2

    try:
        import soundfile  # noqa: F401
        import torch
    except ImportError as exc:
        print(f"Missing Python dependency: {exc}. Run setup_and_run.sh.", file=sys.stderr)
        return 2

    if args.device == "cpu":
        torch.set_num_threads(min(os.cpu_count() or 4, 12))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    disk_free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    print(f"Free disk space near output: {disk_free_gb:.1f} GiB", flush=True)
    if disk_free_gb < 5.0:
        print("Less than 5 GiB is free; the model download may run out of disk space.", file=sys.stderr)
    cache_free_gb = shutil.disk_usage(cache_home).free / (1024**3)
    print(f"Free disk space near Hugging Face model cache: {cache_free_gb:.1f} GiB", flush=True)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(
            f"System RAM: {memory.available / 1024**3:.1f}/{memory.total / 1024**3:.1f} GiB available",
            flush=True,
        )
    except ImportError:
        pass
    progress_path = output_dir / "progress.jsonl"
    run_id = str(uuid.uuid4())

    print(f"Input: {input_path}", flush=True)
    print(f"Narrations: {len(texts)} | output: {output_dir}", flush=True)
    print(f"CPU: {os.cpu_count() or 'unknown'} logical cores", flush=True)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}", flush=True)

    try:
        device, device_index = choose_device(torch, args.device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    voice_description = voice_path.read_text(encoding="utf-8-sig").strip()
    run_settings_hash = hashlib.sha256(
        json.dumps(
            {
                "model": args.model,
                "voice": voice_description,
                "ending_silence": args.ending_silence,
                "ending_fade_ms": args.ending_fade_ms,
                "max_token_cap": args.max_token_cap,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:8]
    records: list[Narration] = []
    for number, text in enumerate(texts, start=1):
        budget = estimate_token_budget(text, args.max_token_cap)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        records.append(
            Narration(
                number,
                text,
                budget,
                output_dir / f"{number:04d}_{slug(text)}_{text_hash}_{run_settings_hash}.wav",
            )
        )

    pending: list[Narration] = []
    for item in records:
        if item.output_path.exists() and not args.force:
            try:
                import soundfile as sf
                info = sf.info(item.output_path)
                if info.frames > 0:
                    print(f"SKIP {item.number}/{len(records)} already exists: {item.output_path.name}", flush=True)
                    append_progress(progress_path, {
                        "run_id": run_id, "time": datetime.now(timezone.utc).isoformat(),
                        "item": item.number, "status": "skipped_existing", "output": str(item.output_path),
                    })
                    continue
            except Exception:
                pass
        pending.append(item)

    if not pending:
        print("All requested WAVs already exist; nothing to generate.", flush=True)
        return 0

    started = time.monotonic()
    try:
        model = load_model(torch, args.model, device)
    except Exception as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2

    if requested_batch_size is None:
        typical_budget = sorted(item.token_budget for item in pending)[len(pending) // 2]
        batch_size = auto_batch_size(torch, device_index, typical_budget, args.max_batch_size)
        if device_index is not None:
            free_gb, total_gb = gpu_summary(torch, device_index)
            print(f"GPU after model load: {free_gb:.1f}/{total_gb:.1f} GiB free", flush=True)
        print(f"Resource-based batch size: {batch_size}", flush=True)
    else:
        batch_size = min(requested_batch_size, args.max_batch_size)
        print(f"Requested batch size: {batch_size} (OOM fallback enabled)", flush=True)

    completed_count = len(records) - len(pending)
    failed_count = 0
    while pending:
        group = pending[:batch_size]
        group_token_cap = max(item.token_budget for item in group)
        texts_batch = [item.text for item in group]
        prompts = [voice_description] * len(group)
        languages = ["English"] * len(group)
        try:
            print(
                f"Generating batch of {len(group)} narration(s), max_new_tokens={group_token_cap}...",
                flush=True,
            )
            wavs, sample_rate = model.generate_voice_design(
                text=texts_batch,
                language=languages,
                instruct=prompts,
                max_new_tokens=group_token_cap,
            )
            if len(wavs) != len(group):
                raise RuntimeError(f"Qwen returned {len(wavs)} WAV(s) for a batch of {len(group)}.")
        except Exception as exc:
            was_oom = is_oom_error(exc, torch)
            if was_oom and device_index is not None:
                torch.cuda.empty_cache()
            if len(group) > 1:
                batch_size = max(1, min(batch_size - 1, len(group) // 2))
                print(
                    f"Batch of {len(group)} failed ({exc}). Retrying smaller at batch size {batch_size}.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            item = group[0]
            pending.pop(0)
            failed_count += 1
            print(f"FAILED {item.number}/{len(records)}: {exc}", file=sys.stderr, flush=True)
            append_progress(progress_path, {
                "run_id": run_id, "time": datetime.now(timezone.utc).isoformat(),
                "item": item.number, "status": "failed", "error": str(exc),
            })
            if was_oom:
                print(
                    "Stopping after a single-item GPU out-of-memory error. Completed WAVs are safe; "
                    "free more VRAM or use a larger GPU, then rerun to resume.",
                    file=sys.stderr,
                    flush=True,
                )
                break
            continue

        for item, wav in zip(group, wavs):
            try:
                item_cap = group_token_cap
                item_wav = wav
                item_sample_rate = sample_rate
                token_retry_oom = False
                while True:
                    estimated_audio_s = len(item_wav) / item_sample_rate
                    cap_seconds = item_cap / CODEC_TOKENS_PER_SECOND
                    likely_truncated = estimated_audio_s >= cap_seconds * 0.90
                    if not likely_truncated or item_cap >= args.max_token_cap:
                        if likely_truncated:
                            print(
                                f"WARNING {item.number}: reached the hard token cap; inspect the ending for truncation.",
                                file=sys.stderr,
                                flush=True,
                            )
                        break
                    next_cap = min(args.max_token_cap, max(item_cap + 512, item_cap * 2))
                    print(
                        f"Item {item.number} reached its {item_cap}-token limit; regenerating with {next_cap} tokens to protect the final words.",
                        flush=True,
                    )
                    try:
                        retried, item_sample_rate = model.generate_voice_design(
                            text=[item.text], language=["English"], instruct=[voice_description],
                            max_new_tokens=next_cap,
                        )
                    except Exception as exc:
                        if is_oom_error(exc, torch):
                            if device_index is not None:
                                torch.cuda.empty_cache()
                            print(
                                f"WARNING {item.number}: extension retry ran out of GPU memory; "
                                "saving the first take. Inspect its ending.",
                                file=sys.stderr,
                                flush=True,
                            )
                            token_retry_oom = True
                            break
                        raise
                    item_wav = retried[0]
                    item_cap = next_cap
                duration = save_audio(
                    item.output_path, item_wav, item_sample_rate,
                    args.ending_fade_ms, args.ending_silence,
                )
                completed_count += 1
                if device_index is not None:
                    free_gb, total_gb = gpu_summary(torch, device_index)
                    resource_note = f"GPU free {free_gb:.1f}/{total_gb:.1f} GiB"
                else:
                    resource_note = "CPU"
                print(
                    f"SAVED {completed_count}/{len(records)}: {item.output_path.name} "
                    f"({duration:.1f}s including ending tail; {resource_note})",
                    flush=True,
                )
                append_progress(progress_path, {
                    "run_id": run_id, "time": datetime.now(timezone.utc).isoformat(),
                    "item": item.number, "status": "saved", "output": str(item.output_path),
                    "duration_s": round(duration, 3), "sample_rate": int(item_sample_rate),
                    "batch_size": len(group), "token_cap": item_cap,
                    "token_retry_oom": token_retry_oom,
                    "device": device,
                })
            except Exception as exc:
                failed_count += 1
                print(f"SAVE FAILED {item.number}/{len(records)}: {exc}", file=sys.stderr, flush=True)
                append_progress(progress_path, {
                    "run_id": run_id, "time": datetime.now(timezone.utc).isoformat(),
                    "item": item.number, "status": "failed_save", "error": str(exc),
                })

        del wavs
        pending = pending[len(group):]
        if device_index is not None:
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    state = "Stopped early" if pending else "Finished"
    print(
        f"{state}: {completed_count}/{len(records)} ready, {failed_count} failed; "
        f"inference time {elapsed / 60:.1f} min. Progress log: {progress_path}",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
