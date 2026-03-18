"""
Fixed evaluation, dataset generation, and utilities for U-Net shape segmentation.
DO NOT MODIFY THIS FILE — it is the ground truth for evaluation and data.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import segmentation_models_pytorch as smp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES = 5  # 0=bg, 1=circle, 2=rectangle, 3=triangle, 4=ellipse
IMG_SIZE = 256
CLASS_NAMES = ["background", "circle", "rectangle", "triangle", "ellipse"]
CLASS_COLORS = np.array(
    [
        [0, 0, 0],  # background — black
        [230, 25, 75],  # circle — red
        [60, 180, 75],  # rectangle — green
        [255, 225, 25],  # triangle — yellow
        [0, 130, 200],  # ellipse — blue
    ],
    dtype=np.uint8,
)
BASE_HUES = [0, 120, 60, 210]  # circle=red, rect=green, tri=yellow, ellipse=blue

TRAIN_SAMPLES = 400
VAL_SAMPLES = 100
TRAIN_SEED_OFFSET = 0
VAL_SEED_OFFSET = 10000

TIME_BUDGET = 300  # 5 minutes wall clock for training


# ── Synthetic dataset ────────────────────────────────────────────────────────
def _rand_color_for_class(cls_idx):
    """Random color with hue near the class base hue, varied saturation/value."""
    base_h = BASE_HUES[cls_idx]
    h = (base_h + random.randint(-30, 30)) % 180
    s = random.randint(100, 255)
    v = random.randint(100, 255)
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])  # RGB


def _rand_bg(img):
    """Gradient or solid background with random tint."""
    mode = random.choice(["solid", "gradient_h", "gradient_v", "gradient_d"])
    c1 = [random.randint(10, 120) for _ in range(3)]
    c2 = [random.randint(10, 120) for _ in range(3)]
    if mode == "solid":
        img[:] = c1
    else:
        for i in range(IMG_SIZE):
            t = i / IMG_SIZE
            c = [int(c1[j] * (1 - t) + c2[j] * t) for j in range(3)]
            if mode == "gradient_h":
                img[:, i] = c
            elif mode == "gradient_v":
                img[i, :] = c
            else:  # diagonal
                img[i, :] = [int(c1[j] * (1 - t) + c2[j] * t) for j in range(3)]


def _draw_circle(img, mask, cls=1):
    cx = random.randint(-20, IMG_SIZE + 20)
    cy = random.randint(-20, IMG_SIZE + 20)
    r = random.randint(8, 60)
    color = _rand_color_for_class(cls - 1)
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(mask, (cx, cy), r, cls, -1)


def _draw_rectangle(img, mask, cls=2):
    x1 = random.randint(-30, IMG_SIZE - 20)
    y1 = random.randint(-30, IMG_SIZE - 20)
    w = random.randint(15, 100)
    h = random.randint(15, 100)
    angle = random.randint(0, 90)
    color = _rand_color_for_class(cls - 1)
    rect_pts = cv2.boxPoints(((x1 + w / 2, y1 + h / 2), (w, h), angle))
    rect_pts = np.intp(rect_pts)
    cv2.fillPoly(img, [rect_pts], color)
    cv2.fillPoly(mask, [rect_pts], cls)


def _draw_triangle(img, mask, cls=3):
    cx = random.randint(-20, IMG_SIZE + 20)
    cy = random.randint(-20, IMG_SIZE + 20)
    size = random.randint(12, 65)
    angle = np.radians(random.randint(0, 360))
    pts = []
    for k in range(3):
        a = angle + k * 2 * 3.14159 / 3 + random.uniform(-0.3, 0.3)
        pts.append([int(cx + size * np.cos(a)), int(cy + size * np.sin(a))])
    pts = np.array(pts, dtype=np.int32)
    color = _rand_color_for_class(cls - 1)
    cv2.fillPoly(img, [pts], color)
    cv2.fillPoly(mask, [pts], cls)


def _draw_ellipse(img, mask, cls=4):
    cx = random.randint(-20, IMG_SIZE + 20)
    cy = random.randint(-20, IMG_SIZE + 20)
    ax1 = random.randint(10, 65)
    ax2 = random.randint(6, 45)
    angle = random.randint(0, 180)
    color = _rand_color_for_class(cls - 1)
    cv2.ellipse(img, (cx, cy), (ax1, ax2), angle, 0, 360, color, -1)
    cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, cls, -1)


DRAW_FNS = [_draw_circle, _draw_rectangle, _draw_triangle, _draw_ellipse]


def generate_sample(rng_seed=None):
    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed % (2**31))

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _rand_bg(img)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    n_shapes = random.randint(4, 10)
    for _ in range(n_shapes):
        fn = random.choice(DRAW_FNS)
        fn(img, mask)

    noise = np.random.randn(*img.shape) * random.uniform(10, 30)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img, mask


class ShapesDataset(Dataset):
    def __init__(self, n_samples, seed_offset=0):
        self.n = n_samples
        self.offset = seed_offset
        self.data = []
        for i in range(n_samples):
            img, mask = generate_sample(rng_seed=seed_offset + i)
            self.data.append((img, mask))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask


# ── Losses ───────────────────────────────────────────────────────────────────
def build_losses():
    return {
        "ce": nn.CrossEntropyLoss(),
        "dice": smp.losses.DiceLoss(mode="multiclass"),
        "jaccard": smp.losses.JaccardLoss(mode="multiclass"),
        "focal": smp.losses.FocalLoss(mode="multiclass"),
        "lovasz": smp.losses.LovaszLoss(mode="multiclass"),
    }


def safe_loss(fn, logits, masks):
    try:
        return fn(logits, masks).item()
    except (ValueError, RuntimeError):
        return fn(logits.cpu(), masks.cpu()).item()


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_miou(pred, target, num_classes=NUM_CLASSES):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


# ── Visualization ────────────────────────────────────────────────────────────
def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def mask_to_rgb(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask_np == c] = CLASS_COLORS[c]
    return rgb


def plot_loss_curves(history, losses_dict, output_path="output/loss_curves.png"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = list(losses_dict.keys()) + ["miou"]
    for ax, m in zip(axes.flat, metrics):
        ax.plot(history[m]["train"], label="train")
        ax.plot(history[m]["val"], label="val")
        ax.set_title(m.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("U-Net on Synthetic Shapes — All Losses", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curves → {output_path}")


def plot_inference(
    model, val_ds, device, n_examples=5, output_path="output/inference_examples.png"
):
    model.eval()
    n_examples = min(n_examples, len(val_ds))
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
    for row in range(n_examples):
        img, gt_mask = val_ds[row]
        with torch.no_grad():
            pred = (
                model(img.unsqueeze(0).to(device)).argmax(1).squeeze(0).cpu().numpy()
            )
        axes[row, 0].imshow(denorm(img))
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(mask_to_rgb(gt_mask.numpy()))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(mask_to_rgb(pred))
        axes[row, 2].set_title("Prediction")
        for c in range(3):
            axes[row, c].axis("off")
    fig.suptitle("Inference Examples (Validation Set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved inference examples → {output_path}")


def evaluate(model, val_loader, losses_dict, device):
    """Full evaluation pass — returns dict of loss values and mIoU."""
    model.eval()
    epoch_losses = {k: 0.0 for k in losses_dict}
    epoch_miou = 0.0
    n_val = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            for k, fn in losses_dict.items():
                epoch_losses[k] += safe_loss(fn, logits, masks) * imgs.size(0)
            preds = logits.argmax(1)
            epoch_miou += compute_miou(preds, masks) * imgs.size(0)
            n_val += imgs.size(0)
    results = {k: v / n_val for k, v in epoch_losses.items()}
    results["miou"] = epoch_miou / n_val
    return results
