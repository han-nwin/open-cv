"""Assignment 4 - Image Segmentation.

Task 1: K-means clustering (K=4, from-scratch NumPy, two random seeds).
Task 2: FCN with pretrained ResNet-50 encoder, trained on 20 VOC pairs.

Run:  python main.py
Outputs are written under ./results/.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------- Paths / constants ----------
ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "segmentation_data"
TEST_DIR = ROOT / "segmentation_data" / "testing-dataset"
OUT_DIR = ROOT / "results"
OUT_KMEANS = OUT_DIR / "task1_kmeans"
OUT_UNET = OUT_DIR / "task2_unet"

N_CLASSES = 21
IGNORE_INDEX = 255
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------- VOC palette & dataset verification ----------
def voc_palette_flat() -> list[int]:
    """Canonical 768-int flat palette used by Pascal VOC indexed masks."""
    palette = [0] * (256 * 3)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i * 3 : i * 3 + 3] = [r, g, b]
    return palette


def voc_palette_rgb() -> np.ndarray:
    return np.array(voc_palette_flat(), dtype=np.uint8).reshape(256, 3)


def verify_voc_dataset(dirs: list[Path]) -> None:
    """Hard-fail if any image/mask pair isn't a valid Pascal VOC indexed mask."""
    voc = voc_palette_flat()
    print("[verify] Checking dataset is in Pascal VOC format...")
    for d in dirs:
        jpgs = sorted(d.glob("*.jpg"))
        assert jpgs, f"No .jpg files found in {d}"
        for jpg in jpgs:
            png = jpg.with_suffix(".png")
            assert png.exists(), f"Missing mask: {png}"
            img = Image.open(jpg)
            mask = Image.open(png)
            assert img.size == mask.size, f"Size mismatch: {jpg.name}"
            assert mask.mode == "P", f"{png.name}: expected mode 'P', got {mask.mode!r}"
            pal = mask.getpalette()
            assert pal is not None, f"{png.name}: no palette"
            # Check first 21 class colors match VOC palette
            assert pal[: 21 * 3] == voc[: 21 * 3], f"{png.name}: palette != VOC"
            arr = np.array(mask)
            assert (
                (arr <= 20) | (arr == 255)
            ).all(), f"{png.name}: out-of-range class id"
        print(f"   {d.relative_to(ROOT)}: {len(jpgs)} pairs OK")
    print("[verify] All pairs are valid Pascal VOC indexed masks.\n")


# ---------- Task 1: K-means from scratch ----------
def kmeans_numpy(
    X: np.ndarray, K: int, seed: int, max_iter: int = 30, tol: float = 1e-3
):
    """Vanilla K-means. X: (N, D) float32. Returns (labels (N,), centers (K, D))."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    init_idx = rng.choice(N, size=K, replace=False)
    centers = X[init_idx].copy()
    labels = np.zeros(N, dtype=np.int64)
    for _ in range(max_iter):
        # assignment step: squared L2 distance to each center
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        # update step
        new_centers = centers.copy()
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                # empty cluster: reseed from a random point
                new_centers[k] = X[rng.integers(0, N)]
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break
    return labels, centers


def kmeans_segment_rgb(img_rgb: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Return segmented image, coloring each pixel with its cluster's mean RGB."""
    H, W, _ = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    labels, centers = kmeans_numpy(X, K, seed)
    seg = centers[labels].reshape(H, W, 3)
    return (seg * 255.0).clip(0, 255).astype(np.uint8)


def run_task1(K: int = 4, seeds: tuple[int, int] = (0, 42)) -> None:
    print(f"[task1] K-means segmentation  K={K}  seeds={seeds}")
    jpgs = sorted(TRAIN_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.jpg"))
    OUT_KMEANS.mkdir(parents=True, exist_ok=True)
    for i, jpg in enumerate(jpgs, 1):
        stem = jpg.stem
        img = np.array(Image.open(jpg).convert("RGB"))
        seg1 = kmeans_segment_rgb(img, K, seeds[0])
        seg2 = kmeans_segment_rgb(img, K, seeds[1])
        out = OUT_KMEANS / stem
        out.mkdir(exist_ok=True)
        Image.fromarray(img).save(out / "original.png")
        Image.fromarray(seg1).save(out / "seed1_segmented.png")
        Image.fromarray(seg2).save(out / "seed2_segmented.png")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("original")
        axes[0].axis("off")
        axes[1].imshow(seg1)
        axes[1].set_title(f"K-means seed={seeds[0]}")
        axes[1].axis("off")
        axes[2].imshow(seg2)
        axes[2].set_title(f"K-means seed={seeds[1]}")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(out / "comparison.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"   [{i:2d}/{len(jpgs)}] {stem}")
    print(f"[task1] Results written to {OUT_KMEANS}\n")


# ---------- Task 2: FCN with pretrained ResNet-50 encoder ----------
class FCNResnet50(nn.Module):
    """FCN-ResNet50 from torchvision.

    Backbone (ResNet-50) is initialized from ImageNet classification weights.
    Segmentation head (FCNHead) is trained from scratch on our 20 training images.
    Forward returns raw logits at the input resolution, shape (B, n_classes, H, W).
    """

    def __init__(self, n_classes: int = 21):
        super().__init__()
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import fcn_resnet50

        self.net = fcn_resnet50(
            weights=None,  # no segmentation-head pretraining
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,  # ImageNet backbone
            num_classes=n_classes,
            aux_loss=False,
        )

    def forward(self, x):
        return self.net(x)["out"]

    def param_groups(self, encoder_lr: float, decoder_lr: float):
        return [
            {"params": self.net.backbone.parameters(), "lr": encoder_lr},
            {"params": self.net.classifier.parameters(), "lr": decoder_lr},
        ]


class SegDataset(Dataset):
    def __init__(self, data_dir: Path, size: int = IMAGE_SIZE, augment: bool = False):
        self.jpgs = sorted(data_dir.glob("*.jpg"))
        self.size = size
        self.augment = augment
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.jpgs)

    def __getitem__(self, idx):
        jpg = self.jpgs[idx]
        img = Image.open(jpg).convert("RGB")
        mask = Image.open(jpg.with_suffix(".png"))  # mode P, class-id values

        if self.augment:
            pre = self.size + 32
            img = img.resize((pre, pre), Image.Resampling.BILINEAR)
            mask = mask.resize((pre, pre), Image.Resampling.NEAREST)
            left = random.randint(0, pre - self.size)
            top = random.randint(0, pre - self.size)
            box = (left, top, left + self.size, top + self.size)
            img = img.crop(box)
            mask = mask.crop(box)
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(img)
        else:
            img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)

        img_t = self.normalize(transforms.functional.to_tensor(img))
        mask_t = torch.from_numpy(np.array(mask)).long()
        return img_t, mask_t, jpg.stem


def train_model(epochs: int, batch_size: int, encoder_lr: float, decoder_lr: float):
    print(f"[task2] Training on device={DEVICE}")
    train_ds = SegDataset(TRAIN_DIR, size=IMAGE_SIZE, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    model = FCNResnet50(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.param_groups(encoder_lr, decoder_lr))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    losses: list[float] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n = 0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        avg = ep_loss / n
        losses.append(avg)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(
                f"   epoch {ep:3d}/{epochs}  loss={avg:.4f}  elapsed={time.time()-t0:.0f}s"
            )

    OUT_UNET.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg cross-entropy loss")
    ax.set_title("FCN-ResNet50 training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_UNET / "train_loss.png", dpi=100)
    plt.close(fig)
    return model, losses


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return voc_palette_rgb()[mask]


@torch.no_grad()
def evaluate_model(model: FCNResnet50) -> None:
    print("[task2] Evaluating on test set")
    model.eval()
    test_ds = SegDataset(TEST_DIR, size=IMAGE_SIZE, augment=False)
    palette_flat = voc_palette_flat()
    OUT_UNET.mkdir(parents=True, exist_ok=True)

    total_correct = 0
    total_valid = 0
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    per_image_lines: list[str] = []

    for idx in range(len(test_ds)):
        img_t, _, stem = test_ds[idx]
        jpg = test_ds.jpgs[idx]
        orig_img = np.array(Image.open(jpg).convert("RGB"))
        orig_mask = np.array(Image.open(jpg.with_suffix(".png")))
        H, W = orig_mask.shape

        logits = model(img_t.unsqueeze(0).to(DEVICE))
        pred_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        # Resize prediction back to original resolution (nearest preserves class ids)
        pred_pil = Image.fromarray(pred_small, mode="P")
        pred_pil.putpalette(palette_flat)
        pred_pil = pred_pil.resize((W, H), Image.Resampling.NEAREST)
        pred = np.array(pred_pil)

        # Save predicted mask (indexed PNG with VOC palette)
        out_pred = Image.fromarray(pred, mode="P")
        out_pred.putpalette(palette_flat)
        out_pred.save(OUT_UNET / f"{stem}_pred.png")

        # Overlay: alpha-blend predicted color over original
        pred_rgb = colorize_mask(pred)
        overlay = (0.5 * pred_rgb + 0.5 * orig_img).clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(OUT_UNET / f"{stem}_overlay.png")

        # Side-by-side compare figure
        gt_rgb = colorize_mask(orig_mask)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title("original")
        axes[0].axis("off")
        axes[1].imshow(gt_rgb)
        axes[1].set_title("ground truth")
        axes[1].axis("off")
        axes[2].imshow(pred_rgb)
        axes[2].set_title("prediction")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(OUT_UNET / f"{stem}_compare.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Metrics (ignore boundary=255)
        valid = orig_mask != IGNORE_INDEX
        correct = (pred == orig_mask) & valid
        n_valid = int(valid.sum())
        pix_acc = correct.sum() / max(1, n_valid)
        total_correct += int(correct.sum())
        total_valid += n_valid

        ious: list[float] = []
        for c in range(N_CLASSES):
            gt_c = orig_mask == c
            pr_c = (pred == c) & valid
            union = int((gt_c | pr_c).sum())
            if union == 0:
                continue
            inter = int((gt_c & pr_c).sum())
            ious.append(inter / union)
        miou = float(np.mean(ious)) if ious else 0.0
        per_image_lines.append(f"{stem}  pix_acc={pix_acc:.3f}  mIoU={miou:.3f}")
        print(f"   {stem}  pix_acc={pix_acc:.3f}  mIoU={miou:.3f}")

        # Global confusion for per-class IoU
        for c in range(N_CLASSES):
            gt_c = (orig_mask == c) & valid
            for p in range(N_CLASSES):
                conf[c, p] += int((gt_c & (pred == p)).sum())

    overall_acc = total_correct / max(1, total_valid)
    per_class_iou: list[float | None] = []
    for c in range(N_CLASSES):
        tp = int(conf[c, c])
        fp = int(conf[:, c].sum()) - tp
        fn = int(conf[c, :].sum()) - tp
        denom = tp + fp + fn
        per_class_iou.append(None if denom == 0 else tp / denom)
    defined = [v for v in per_class_iou if v is not None]
    mean_iou = float(np.mean(defined)) if defined else 0.0

    lines = [
        "Per-image metrics:",
        *per_image_lines,
        "",
        f"Overall pixel accuracy: {overall_acc:.4f}",
        f"Mean IoU (present classes): {mean_iou:.4f}",
        "",
        "Per-class IoU:",
    ]
    for c, v in enumerate(per_class_iou):
        tag = "n/a (absent)" if v is None else f"{v:.4f}"
        lines.append(f"   {c:2d} {VOC_CLASSES[c]:14s}  {tag}")
    lines += [
        "",
        "Caveats:",
        "  * 20 training images -> strong overfitting risk.",
        "  * Test set contains classes (8 cat, 10 cow, 19 train) never seen in training.",
        "  * Goal is methodology comparison (Task 1 vs Task 2), not SOTA.",
    ]
    (OUT_UNET / "metrics.txt").write_text("\n".join(lines))
    print(f"[task2] Metrics -> {OUT_UNET / 'metrics.txt'}")
    print(f"[task2] Overall pixel accuracy: {overall_acc:.4f}")
    print(f"[task2] Mean IoU: {mean_iou:.4f}\n")


def run_task2(
    epochs: int = 80,
    batch_size: int = 4,
    encoder_lr: float = 1e-4,
    decoder_lr: float = 1e-3,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model, _ = train_model(
        epochs=epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
    )
    evaluate_model(model)


# ---------- Main ----------
if __name__ == "__main__":
    verify_voc_dataset([TRAIN_DIR, TEST_DIR])
    run_task1(K=4, seeds=(0, 42))
    run_task2(epochs=50)
