#!/usr/bin/env python3
"""Generate instrumental variations through ACE-Step 1.5's native batch API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "prompt.txt"
DEFAULT_OUTPUT = HERE / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--duration", type=float, default=60.0, help="Target audio duration in seconds (10–600)")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducible variations")
    parser.add_argument("--dit-model", default=os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"))
    parser.add_argument("--lm-model", default=os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 10 <= args.duration <= 600:
        parser.error("--duration must be between 10 and 600 seconds")
    return args


def progress_record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    args = parse_args()
    prompt_path = args.prompt.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not prompt_path.is_file():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        return 2
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        print(f"Prompt file is empty: {prompt_path}", file=sys.stderr)
        return 2

    try:
        import torch
        from acestep.model_downloader import ensure_lm_model
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
    except Exception as exc:
        print("ACE-Step dependencies could not be imported. Run setup_and_run.sh first.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not visible to PyTorch; stopping instead of using CPU on a rented GPU.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    run_started = time.monotonic()
    print(f"Prompt: {prompt_path}", flush=True)
    print(f"Batch requested: {args.batch_size}; duration: {args.duration:g}s", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GiB", flush=True)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(HERE / ".vendor" / "ACE-Step-1.5"),
        config_path=args.dit_model,
        device=args.device,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    print(status, flush=True)
    if not ok:
        return 1

    lm = LLMHandler()
    checkpoint_dir = HERE / ".vendor" / "ACE-Step-1.5" / "checkpoints"
    print(f"Ensuring language model is available: {args.lm_model}", flush=True)
    lm_ok, lm_message = ensure_lm_model(args.lm_model, checkpoint_dir)
    print(lm_message, flush=True)
    if not lm_ok:
        return 1
    status, ok = lm.initialize(
        checkpoint_dir=str(checkpoint_dir),
        lm_model_path=args.lm_model,
        backend="pt",
        device=args.device,
        offload_to_cpu=False,
    )
    print(status, flush=True)
    if not ok:
        return 1

    params = GenerationParams(
        task_type="text2music",
        caption=prompt,
        lyrics="[Instrumental]",
        instrumental=True,
        bpm=65,
        duration=args.duration,
        inference_steps=8,
        shift=3.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )

    batch_size = args.batch_size
    while batch_size >= 1:
        seeds = None if args.seed is None else list(range(args.seed, args.seed + batch_size))
        config = GenerationConfig(
            batch_size=batch_size,
            allow_lm_batch=False,
            use_random_seed=args.seed is None,
            seeds=seeds,
            lm_batch_chunk_size=1,
            audio_format="flac",
        )
        print(f"Generating {batch_size} variation(s) with ACE-Step native batch...", flush=True)
        started = time.monotonic()
        result = None
        try:
            result = generate_music(dit, lm, params, config, save_dir=str(output_dir))
            if not result.success:
                raise RuntimeError(result.error or result.status_message or "ACE-Step generation failed")
            audios = result.audios or []
            if not audios:
                raise RuntimeError("ACE-Step reported success but returned no audio files")
            for index, audio in enumerate(audios, 1):
                source = Path(audio["path"])
                seed = audio.get("params", {}).get("seed")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                target = output_dir / f"piano_{stamp}_b{batch_size}_{index:02d}_seed{seed if seed is not None else 'random'}.flac"
                if source.resolve() != target.resolve():
                    if target.exists():
                        target = target.with_name(f"{target.stem}_{time.time_ns()}{target.suffix}")
                    source.replace(target)
                record = {
                    "status": "saved",
                    "path": str(target),
                    "seed": seed,
                    "batch_size": batch_size,
                    "duration_seconds": args.duration,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                progress_record(progress_path, record)
                print(f"SAVED {index}/{len(audios)}: {target.name} (seed={seed})", flush=True)
            print(f"Completed {len(audios)} variation(s) in {time.monotonic() - started:.1f}s", flush=True)
            print(f"Total process time including model load: {time.monotonic() - run_started:.1f}s", flush=True)
            return 0
        except Exception as exc:
            message = str(exc)
            oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_oom = "out of memory" in message.lower() or (oom_type is not None and isinstance(exc, oom_type))
            if not is_oom or batch_size == 1:
                progress_record(progress_path, {
                    "status": "failed",
                    "error": message,
                    "batch_size": batch_size,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                print(f"Generation failed at batch size {batch_size}: {message}", file=sys.stderr, flush=True)
                return 1
            next_size = max(1, batch_size // 2)
            print(f"CUDA out of memory at batch {batch_size}; clearing cache and retrying with batch {next_size}.", flush=True)
            progress_record(progress_path, {
                "status": "oom_retry",
                "batch_size": batch_size,
                "next_batch_size": next_size,
                "error": message,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            if result is not None:
                del result
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            batch_size = next_size

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

