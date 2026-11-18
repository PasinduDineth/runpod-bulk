# Qwen3-TTS bulk generator

This folder is a standalone bulk narration tool for a rented Linux GPU. It installs an isolated Python environment, installs a CUDA-compatible PyTorch wheel when an NVIDIA GPU is detected, installs Qwen3-TTS, and downloads the model automatically on first run.

## Use it

1. Clone this repository on the rented machine:

   ```bash
   git clone https://github.com/PasinduDineth/runpod-bulk.git
   cd runpod-bulk
   ```

2. Put one narration in each paragraph of `narrations.txt`. Separate entries with a blank line. Keep paragraph breaks inside a narration only if they should be separate clips.
3. The reference voice is included as `clone.mp3`. Run the generator:

   ```bash
   bash setup_and_run.sh
   ```

The default is the accepted voice-clone setup: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with `clone.mp3`, native speech rate (`1.0`), 0.6 seconds between sentences, and no added comma pauses. It reads `narrations.txt` and saves WAVs in `output-voice-clone/`, with a progress record after each narration. The setup installs the isolated environment and downloads model weights automatically on first run. Later runs in the same pod reuse `.venv/` and `.cache/huggingface/`.

## Reference voice cloning with pacing controls

To use the alternative VoiceDesign mode and `voice_description.txt`:

```bash
QWEN_TTS_MODE=voice-design bash setup_and_run.sh
```

Clone mode reuses one speaker prompt from the committed `clone.mp3` for the whole run. The generator splits at existing sentence and comma punctuation, then inserts the configured silence between phrases. With the saved defaults, commas remain contiguous and only sentence boundaries receive a pause. It does not require ellipses or pause markup in `narrations.txt`.

Qwen's Base voice-clone API has no native speech-rate parameter. `--speed` is our generator's pitch-preserving post-processing control: `1.0` keeps the generated speech rate, while `0.85` makes the spoken phrases about 18% longer without stretching the inserted pauses. `--sentence-pause` and `--comma-pause` set the inserted silence lengths in seconds. The saved defaults are `--speed 1.0 --sentence-pause 0.6 --comma-pause 0`. Phrase-by-phrase cloning takes longer than one batch call, but keeps pause timing independent from speaking rate.

For a new rented pod, cloning the repository and running `bash setup_and_run.sh` is sufficient. The pod needs network access for package/model downloads. Pod-local `.venv/`, `.cache/huggingface/`, and output files are not in Git; they are recreated on a fresh pod. To avoid downloading the model again, mount persistent storage and point `HF_HOME` at a directory on that volume before running. Keep the reference audio and narration text in Git; generated audio stays in the output folder and is not committed by default.

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

