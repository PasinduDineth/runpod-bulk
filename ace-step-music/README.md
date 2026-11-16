# ACE-Step background music generator

This separate folder prepares ACE-Step 1.5 on a rented Linux GPU and generates instrumental background music. It does not change the Qwen TTS generator.

## First run

On the GPU pod, copy this `ace-step-music` folder, then run:

```bash
cd ace-step-music
bash setup_and_run.sh --quality maximum --count 1
```

The first run installs Git if needed, installs `uv`, clones the official [ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5) into `.vendor/`, syncs its environment, downloads its models, and generates one test track. The model cache and outputs stay in this folder so later runs can reuse them.

The default `maximum` profile uses ACE-Step's SFT model at 50 inference steps and the 1.7B language model with reasoning enabled. It generates tracks **one at a time**, keeping one set of models loaded and saving each result before beginning the next. Generate 10 tracks with:

```bash
bash setup_and_run.sh --quality maximum --count 10 --output output/ten-track-run
```

The maximum profile can take substantially longer than Turbo and requires an extra SFT checkpoint download on first use. The checkpoint is about 4.5 GB. The SFT model is detail-oriented, though ACE-Step notes Turbo can retain slightly better audio clarity. If SFT does not fit the GPU, the run stops with an error rather than silently switching to a different quality profile.

For a faster option that is more suitable for an 8 GB GPU, use:

```bash
bash setup_and_run.sh --quality balanced --count 10 --output output/ten-track-run
```

## Prompt and options

Edit `prompt.txt` to change the musical direction. The supplied prompt asks for instrumental-only, slow neo-classical felt piano and a restrained winter mood. `instrumental=True` and the `[Instrumental]` lyric marker are both set to avoid generated vocals.

```bash
bash setup_and_run.sh --quality maximum --count 10 --duration 60
bash setup_and_run.sh --quality balanced --count 10 --duration 90 --output output
```

Outputs are FLAC files under `output/`, with `progress.jsonl` updated immediately after each saved track. Each record includes track number, selected quality profile, per-track time, and cumulative time. Filenames are timestamped so separate runs do not overwrite earlier tracks. Default duration is 60 seconds and default track count is one.

## GPU notes

- Use a Linux CUDA pod with Python 3.11 or 3.12. The setup installs ACE-Step from its official GitHub source.
- The maximum profile uses the 1.7B LM with the PyTorch backend; on low-free-VRAM GPUs the script offloads the LM to CPU. The balanced profile uses the 0.6B LM.
- After loading the main model, the runner checks free VRAM. If less than 2.5 GiB remains, it places the selected language model on CPU to reserve VRAM for audio generation. This can make the LM stage slower.
- Model initialization and first-run downloads are included in wall-clock time; subsequent generations reuse the downloaded models.
- The script checks that CUDA is available and stops instead of silently using CPU on a rented GPU.

