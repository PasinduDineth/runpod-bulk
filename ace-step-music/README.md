# ACE-Step background music generator

This separate folder prepares ACE-Step 1.5 on a rented Linux GPU and generates instrumental background music. It does not change the Qwen TTS generator.

## First run

On the GPU pod, copy this `ace-step-music` folder, then run:

```bash
cd ace-step-music
bash setup_and_run.sh --batch-size 1
```

The first run installs Git if needed, installs `uv`, clones the official [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5) into `.vendor/`, syncs its environment, downloads its models, and generates one test track. The model cache and outputs stay in this folder so later runs can reuse them.

After reviewing the first track, generate four variations from the same prompt with:

```bash
bash setup_and_run.sh --batch-size 4
```

ACE-Step supports a native batch of up to 8 variations. The script defaults to the 2B Turbo DiT and the 0.6B language model recommended by ACE-Step for 6–8 GB GPUs. Batch 4 can still exceed available VRAM depending on the pod and duration; on CUDA out-of-memory, the script retries at batch 2 and then 1, recording the actual batch size used. This keeps one model copy loaded rather than launching four competing processes.

## Prompt and options

Edit `prompt.txt` to change the musical direction. The supplied prompt asks for instrumental-only, slow neo-classical felt piano and a restrained winter mood. `instrumental=True` and the `[Instrumental]` lyric marker are both set to avoid generated vocals.

```bash
bash setup_and_run.sh --batch-size 1 --duration 60
bash setup_and_run.sh --batch-size 4 --duration 90 --output output
```

Outputs are FLAC files under `output/`, with `progress.jsonl` updated after each saved result. Filenames are timestamped so separate runs do not overwrite earlier tracks. Default duration is 60 seconds.

## GPU notes

- Use a Linux CUDA pod with Python 3.11 or 3.12. The setup installs ACE-Step from its official GitHub source.
- The 0.6B LM uses the PyTorch backend to avoid the extra setup and memory requirements of vLLM on an 8 GB card.
- After loading the main model, the runner checks free VRAM. If less than 2.5 GiB remains, it keeps the 0.6B LM on CPU so the RTX 3070 can reserve VRAM for audio generation. This can make the LM stage slower, but avoids the startup out-of-memory failure seen when both models are placed on an 8 GB GPU.
- Model initialization and first-run downloads are included in wall-clock time; subsequent generations reuse the downloaded models.
- The script checks that CUDA is available and stops instead of silently using CPU on a rented GPU.

