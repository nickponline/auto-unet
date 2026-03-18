"""
U-Net segmentation training on synthetic shapes — MODIFY THIS FILE.
Everything in prepare.py is fixed. You can change anything here:
model architecture, optimizer, scheduler, hyperparameters, training loop, loss.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from prepare import (
    NUM_CLASSES,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    TRAIN_SEED_OFFSET,
    VAL_SEED_OFFSET,
    TIME_BUDGET,
    ShapesDataset,
    build_losses,
    safe_loss,
    compute_miou,
    plot_loss_curves,
    plot_inference,
    evaluate,
)

# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
LR = 1e-3
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
train_ds = ShapesDataset(TRAIN_SAMPLES, seed_offset=TRAIN_SEED_OFFSET)
val_ds = ShapesDataset(VAL_SAMPLES, seed_offset=VAL_SEED_OFFSET)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# ── Model ────────────────────────────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
)
model = model.to(DEVICE)

# ── Optimizer & scheduler ────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop (time-budgeted) ────────────────────────────────────────────
losses_dict = build_losses()
history = {k: {"train": [], "val": []} for k in losses_dict}
history["miou"] = {"train": [], "val": []}

total_start = time.time()
train_start = time.time()
epoch = 0

while True:
    epoch += 1
    elapsed = time.time() - train_start
    if elapsed >= TIME_BUDGET:
        break

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    epoch_losses = {k: 0.0 for k in losses_dict}
    epoch_miou = 0.0
    n_train = 0

    for imgs, masks in train_loader:
        if time.time() - train_start >= TIME_BUDGET:
            break
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)

        loss = losses_dict["ce"](logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for k, fn in losses_dict.items():
                epoch_losses[k] += safe_loss(fn, logits, masks) * imgs.size(0)
            preds = logits.argmax(1)
            epoch_miou += compute_miou(preds, masks) * imgs.size(0)
        n_train += imgs.size(0)

    if n_train == 0:
        break

    for k in losses_dict:
        history[k]["train"].append(epoch_losses[k] / n_train)
    history["miou"]["train"].append(epoch_miou / n_train)

    # ── Val ──────────────────────────────────────────────────────────────
    val_results = evaluate(model, val_loader, losses_dict, DEVICE)
    for k in losses_dict:
        history[k]["val"].append(val_results[k])
    history["miou"]["val"].append(val_results["miou"])

    train_elapsed = time.time() - train_start
    print(
        f"Epoch {epoch:3d} ({train_elapsed:5.1f}s) │ "
        f"CE {history['ce']['train'][-1]:.4f}/{history['ce']['val'][-1]:.4f} │ "
        f"mIoU {history['miou']['train'][-1]:.4f}/{history['miou']['val'][-1]:.4f}"
    )

training_seconds = time.time() - train_start

# ── Save & plot ──────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
torch.save(model.state_dict(), "output/unet_shapes.pth")
plot_loss_curves(history, losses_dict)
plot_inference(model, val_ds, DEVICE)

# ── Final summary ────────────────────────────────────────────────────────────
final = evaluate(model, val_loader, losses_dict, DEVICE)
total_seconds = time.time() - total_start
num_params = sum(p.numel() for p in model.parameters()) / 1e6

print("\n---")
print(f"val_miou:          {final['miou']:.6f}")
print(f"val_ce:            {final['ce']:.6f}")
print(f"val_dice:          {final['dice']:.6f}")
print(f"val_jaccard:       {final['jaccard']:.6f}")
print(f"val_focal:         {final['focal']:.6f}")
print(f"val_lovasz:        {final['lovasz']:.6f}")
print(f"training_seconds:  {training_seconds:.1f}")
print(f"total_seconds:     {total_seconds:.1f}")
print(f"num_epochs:        {epoch}")
print(f"num_params_M:      {num_params:.1f}")
