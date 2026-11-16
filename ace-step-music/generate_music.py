#!/usr/bin/env python3
"""Generate instrumental tracks sequentially through ACE-Step 1.5."""

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
    parser.add_argument("--count", type=int, default=1, help="Number of tracks to generate sequentially (1–100)")
    parser.add_argument(
        "--quality",
        choices=("maximum", "balanced"),
        default="maximum",
        help="maximum uses SFT/50 steps/1.7B LM; balanced uses Turbo/8 steps/0.6B LM",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Target audio duration in seconds (10–600)")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed for reproducible variations")
    parser.add_argument("--dit-model", default=os.environ.get("ACESTEP_CONFIG_PATH"))
    parser.add_argument("--lm-model", default=os.environ.get("ACESTEP_LM_MODEL_PATH"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100")
    if not 10 <= args.duration <= 600:
        parser.error("--duration must be between 10 and 600 seconds")
    profile = {
        "maximum": {
            "dit_model": "acestep-v15-sft",
            "lm_model": "acestep-5Hz-lm-1.7B",
            "inference_steps": 50,
            "guidance_scale": 7.0,
            "shift": 1.0,
        },
        "balanced": {
            "dit_model": "acestep-v15-turbo",
            "lm_model": "acestep-5Hz-lm-0.6B",
            "inference_steps": 8,
            "guidance_scale": 1.0,
            "shift": 3.0,
        },
    }[args.quality]
    args.dit_model = args.dit_model or profile["dit_model"]
    args.lm_model = args.lm_model or profile["lm_model"]
    args.inference_steps = profile["inference_steps"]
    args.guidance_scale = profile["guidance_scale"]
    args.shift = profile["shift"]
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
    print(
        f"Quality: {args.quality} | tracks: {args.count} sequentially | "
        f"duration: {args.duration:g}s | steps: {args.inference_steps}",
        flush=True,
    )
    print(f"DiT: {args.dit_model} | LM: {args.lm_model}", flush=True)
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
    lm_offload_to_cpu = False
    if args.device == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        # The 2B DiT/VAE/text encoder already occupy most of an 8 GB card.
        # Keep at least 2.5 GiB free before placing the additional LM on CUDA.
        lm_offload_to_cpu = free_gib < 2.5
        print(
            f"VRAM after main model load: {free_gib:.2f}/{total_gib:.2f} GiB free; "
            f"language model placement: {'CPU' if lm_offload_to_cpu else 'CUDA'}",
            flush=True,
        )
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
        offload_to_cpu=lm_offload_to_cpu,
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
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        thinking=True,
        use_cot_metas=True,
        use_cot_caption=True,
        use_cot_language=True,
    )

    for track_index in range(args.count):
        track_number = track_index + 1
        seed = None if args.seed is None else args.seed + track_index
        config = GenerationConfig(
            batch_size=1,
            allow_lm_batch=False,
            use_random_seed=seed is None,
            seeds=None if seed is None else [seed],
            lm_batch_chunk_size=1,
            audio_format="flac",
        )
        print(f"Generating track {track_number}/{args.count}...", flush=True)
        started = time.monotonic()
        result = None
        try:
            result = generate_music(dit, lm, params, config, save_dir=str(output_dir))
            if not result.success:
                raise RuntimeError(result.error or result.status_message or "ACE-Step generation failed")
            audios = result.audios or []
            if not audios:
                raise RuntimeError("ACE-Step reported success but returned no audio files")
            if len(audios) != 1:
                raise RuntimeError(f"Expected one sequential result, but ACE-Step returned {len(audios)}")
            for audio in audios:
                source = Path(audio["path"])
                audio_seed = audio.get("params", {}).get("seed")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                target = output_dir / (
                    f"piano_{stamp}_{args.quality}_{track_number:02d}of{args.count:02d}_"
                    f"seed{audio_seed if audio_seed is not None else 'random'}.flac"
                )
                if source.resolve() != target.resolve():
                    if target.exists():
                        target = target.with_name(f"{target.stem}_{time.time_ns()}{target.suffix}")
                    source.replace(target)
                record = {
                    "status": "saved",
                    "path": str(target),
                    "seed": audio_seed,
                    "track_number": track_number,
                    "total_tracks": args.count,
                    "quality": args.quality,
                    "dit_model": args.dit_model,
                    "lm_model": args.lm_model,
                    "inference_steps": args.inference_steps,
                    "duration_seconds": args.duration,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "cumulative_elapsed_seconds": round(time.monotonic() - run_started, 2),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                progress_record(progress_path, record)
                print(
                    f"SAVED {track_number}/{args.count}: {target.name} "
                    f"(seed={audio_seed}; track time={record['elapsed_seconds']:.1f}s; "
                    f"total elapsed={record['cumulative_elapsed_seconds']:.1f}s)",
                    flush=True,
                )
            print(
                f"Track {track_number}/{args.count} generated in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:
            message = str(exc)
            progress_record(progress_path, {
                "status": "failed",
                "track_number": track_number,
                "total_tracks": args.count,
                "quality": args.quality,
                "error": message,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"Track {track_number}/{args.count} failed; stopping the batch: {message}",
                file=sys.stderr,
                flush=True,
            )
            if result is not None:
                del result
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 1

    print(f"All {args.count} tracks completed. Total process time including model load: {time.monotonic() - run_started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

