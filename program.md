# autoresearch — U-Net segmentation

This is an experiment to have the LLM do its own research on U-Net semantic segmentation.

The task: segment synthetic images containing colored shapes (circles, rectangles, triangles, ellipses) on noisy gradient backgrounds. 5 classes total (including background). The dataset is generated procedurally — no external data needed.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, dataset generation, losses, evaluation, plotting. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device (MPS/CUDA/CPU, auto-detected). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup and final evaluation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture (encoder, decoder type), optimizer, hyperparameters, training loop, batch size, learning rate schedule, loss function used for backprop, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, dataset generation, loss definitions, metrics, and constants (time budget, image size, number of classes, etc).
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate` function and `compute_miou` in `prepare.py` are the ground truth metrics.

**The goal is simple: get the highest val_miou.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the encoder (resnet50, efficientnet, etc.), the decoder (Unet, UnetPlusPlus, DeepLabV3+, FPN, etc.), the optimizer, the learning rate, the scheduler, the loss used for backprop, the batch size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_miou:          0.750000
val_ce:            0.234567
val_dice:          0.123456
val_jaccard:       0.345678
val_focal:         0.456789
val_lovasz:        0.567890
training_seconds:  300.1
total_seconds:     315.2
num_epochs:        85
num_params_M:      24.4
```

You can extract the key metric from the log file:

```
grep "^val_miou:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_miou	num_params_M	status	description
```

1. git commit hash (short, 7 chars)
2. val_miou achieved (e.g. 0.750000) — use 0.000000 for crashes
3. num_params_M (e.g. 24.4) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_miou	num_params_M	status	description
a1b2c3d	0.750000	24.4	keep	baseline (resnet34 + Adam 1e-3)
b2c3d4e	0.782000	24.4	keep	switch to dice loss for backprop
c3d4e5f	0.745000	24.4	discard	use SGD instead of Adam
d4e5f6g	0.000000	0.0	crash	resnet152 encoder (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_miou:\|^num_params_M:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_miou improved (higher), you "advance" the branch, keeping the git commit
9. If val_miou is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different encoders, different decoders, loss combinations, learning rate schedules, optimizer tweaks, augmentation-adjacent tricks in the training loop. The loop runs until the human interrupts you, period.

## Ideas to try

Here are some directions worth exploring (not exhaustive):

- **Encoders**: resnet50, efficientnet-b0 through b4, mobilenet_v2, timm-regnety_016, etc.
- **Decoders**: UnetPlusPlus, MAnet, FPN, DeepLabV3Plus, PSPNet, LinkNet
- **Loss for backprop**: dice, jaccard, focal, lovasz, or combinations (e.g. CE + dice)
- **Optimizers**: AdamW, SGD with momentum, RAdam
- **Schedulers**: CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
- **Learning rate**: try 3e-4, 1e-3, 3e-3, etc.
- **Batch size**: larger batches (16, 32) if memory allows — more stable gradients, fewer epochs but bigger steps
- **Encoder freezing**: freeze encoder for first N epochs, then unfreeze
- **Gradient accumulation**: simulate larger batch sizes
