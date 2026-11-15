# Qwen3-TTS bulk generator

This folder is a standalone bulk narration tool for a rented Linux GPU. It installs an isolated Python environment, installs a CUDA-compatible PyTorch wheel when an NVIDIA GPU is detected, installs Qwen3-TTS, and downloads the model automatically on first run.

## Use it

1. Copy this whole `qwen-tts-bulk` folder to the rented machine.
2. Put one narration in each paragraph of `narrations.txt`. Separate entries with a blank line. Keep paragraph breaks inside a narration only if they should be separate clips.
3. Edit `voice_description.txt` if needed.
4. In the folder, run:

   ```bash
   bash setup_and_run.sh
   ```

The default model is `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, so it uses the supplied voice description. Model weights are downloaded into `.cache/huggingface` on the first run and reused later.

## GPU selection and batching

The runner reports visible GPU names and free VRAM, selects the GPU with the most free memory, loads one copy of the model, then chooses a conservative batch size from the VRAM left after model loading. It uses Qwen's list/batch inference API to generate multiple narrations together, avoiding multiple model copies competing for VRAM. If a batch runs out of memory, it automatically retries with a smaller batch. You can override the automatic size with `--batch-size 1`, `--batch-size 2`, and so on; `--max-batch-size` caps auto mode (default 16).

Automatic selection uses one GPU, even if the machine has several. Multiple narrations run together as a batch on that selected GPU; this runner does not load duplicate model copies across multiple cards.

Batch outputs become available together; after the batch returns, the runner saves each WAV and prints its progress line. Set `--batch-size 1` for an update immediately after each individual narration, or lower `--max-batch-size` to trade some throughput for more frequent updates.

If CUDA is not visible to PyTorch, the script stops before model inference rather than silently burning GPU rental time on CPU. To intentionally use CPU, pass `--device cpu`; it will be much slower.

## Saving, resuming, and audio endings

- Each WAV is written as soon as its batch finishes, using a temporary file and atomic rename.
- `output/progress.jsonl` receives a flushed record after every saved, skipped, or failed narration.
- Existing valid WAVs are skipped on the next run, so you can rerun after interruption. Output names include a fingerprint of the narration, model, voice direction, and ending settings, so edits create fresh clips. Pass `--force` to regenerate matching clips.
- The generator adds a final period when needed, gives each item a word-count-based token allowance (up to 8192), checks for likely token-limit truncation, adds a short fade and 1 second of silence after the speech, and saves 24-bit WAV audio. This gives the final word room to resolve instead of ending at the exact speech boundary.

## Options

```bash
bash setup_and_run.sh --batch-size auto --max-batch-size 16
bash setup_and_run.sh --batch-size 2 --force
bash setup_and_run.sh --input narrations.txt --voice voice_description.txt --output output
```

The PyPI install is the default and does not need Git. To install the current upstream source instead, run `QWEN_TTS_SOURCE=git bash setup_and_run.sh`; the bootstrap will install Git if missing (using the system package manager) and then install Qwen from the official repository.

The setup script supports Ubuntu/Debian (`apt`), Fedora/RHEL (`dnf`/`yum`), and Alpine (`apk`) when system packages are missing. Installing system packages may require root or `sudo`.

