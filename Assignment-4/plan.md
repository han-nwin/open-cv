# Assignment 4 — Implementation Plan

Single-file `main.py` running both tasks end-to-end on `segmentation_data/`. PyTorch + NumPy + OpenCV/PIL + Matplotlib only (no sklearn).

## Dataset (already inspected)
- `segmentation_data/*.jpg` + `*.png`: 20 training pairs (image + Pascal VOC indexed mask).
- `segmentation_data/testing-dataset/*.jpg` + `*.png`: 10 test pairs.
- Mask values are class indices 0..20 with 255 as the boundary/ignore label. Pascal VOC color palette is used for visualization.
- Class IDs observed in training: {0,1,3,4,5,6,7,9,13,14,15,16,18,20}; in testing some classes appear that aren't in train (8,10,19) — expected limitation, will be noted in the writeup.

## Submission deliverables (per a4.md)
1. **Source code** — `main.py` (+ `verify.py` for dataset check).
2. **Result images** — everything under `results/` (see tree below).
3. **Report** — `report.md` written by hand, covering both tasks. Content outline further down.

## Output layout
```
Assignment-4/
  main.py                       # source code (both tasks)
  verify.py                     # dataset format sanity check
  report.md                     # written report (Task 1 + Task 2 analysis)
  results/
    task1_kmeans/<image_id>/
      original.png
      seed1_segmented.png       # seed=0
      seed2_segmented.png       # seed=42
      comparison.png            # side-by-side: original | seed1 | seed2
    task2_unet/
      train_loss.png            # training loss curve
      metrics.txt               # per-image + mean pixel-acc + mean IoU
      <image_id>_pred.png       # predicted mask (VOC palette)
      <image_id>_overlay.png    # prediction overlaid on image
      <image_id>_compare.png    # original | gt | pred
```

## Task 1 — K-Means (K=4)

### What this task is doing (plain English)
K-means is **unsupervised** — it never looks at the ground-truth masks. We treat every pixel as just a 3-number color (R, G, B) and ask: "which 4 groups (clusters) of similar colors best summarize this image?"

The algorithm:
1. Drop K=4 random "guess" colors into the color space (these are *cluster centers*).
2. Assign every pixel to whichever guess color it's closest to (Euclidean distance in RGB).
3. Move each guess to the average color of the pixels assigned to it.
4. Repeat 2–3 until the guesses stop moving.

When it converges, every pixel has a label `0`, `1`, `2`, or `3`. Recolor the image with the average color of each cluster and you get a "posterized" version of the original — that's the segmentation.

### Why two seeds
K-means is sensitive to **where you place the initial guesses**. Different starting points → different final clusters → different segmentations of the same image. Running with seeds 0 and 42 demonstrates this instability, which is one of K-means's known weaknesses.

### What it can and can't do
- ✅ Groups visually similar regions (sky, grass, plane body) cheaply, no training needed.
- ❌ Doesn't know what *anything is*. Cluster 2 isn't "airplane" — it's just "the dark-red-ish pixels." If two unrelated objects share a color, K-means lumps them together.
- ❌ Boundaries between clusters are jagged because each pixel is decided in isolation (no spatial context).

### Steps
1. For each image in `segmentation_data/*.jpg`:
   - Load RGB, use **RGB** color space directly.
   - Reshape to (H*W, 3) float32 feature matrix (values scaled to [0,1]).
   - Run K-means **from scratch in NumPy** with K=4:
     - Init: pick K random pixels as initial centers (seeded RNG).
     - Iterate: assign each pixel to nearest center (vectorized L2), recompute centers as mean. Stop when centers move < 1e-3 or 30 iters.
   - Run twice with **seeds 0 and 42** to show initialization sensitivity.
   - Color each pixel by its cluster's mean RGB for display.
2. Save original, two segmentations, and a side-by-side comparison.
3. Brief written analysis embedded as a print summary at end of Task 1 (initialization sensitivity, no semantic awareness, slow on big images, etc.).

## Task 2 — U-Net with pretrained ResNet encoder

### What this task is doing (plain English)
Now we **do** use the masks. The mask PNGs hold a class number (0=background, 1=aeroplane, ..., 20=tvmonitor) for every pixel. Our job is to train a neural network that, given just the JPG, outputs a guess for the class of every pixel — and we judge it against the mask.

This is **supervised** learning: image → predicted class map, compare to ground-truth class map, compute loss, adjust the network's weights via gradient descent, repeat for many epochs.

### Why a U-Net (recap)
- An **encoder** stack of conv+pool layers shrinks the image down. As it shrinks, each "pixel" of the small feature map sees a wider area of the original image, so the network learns *what* is in big regions.
- A **decoder** stack mirrors the encoder, upsampling back to full resolution so we get a label per original pixel.
- **Skip connections** copy the encoder's high-resolution features straight across to the decoder, so it can keep object boundaries sharp instead of producing blobby blurs.

### Why "pretrained ResNet encoder"
Training a network from scratch on 20 images is hopeless — there isn't enough data for it to even learn what edges and textures look like. Trick: take a **ResNet-18 already trained on ImageNet** (1.2 million labeled photos) and use *its* convolutional layers as the encoder. That encoder already knows generic visual concepts — edges, fur, wheels, sky textures — for free. We then bolt on a fresh decoder and only have to teach the model how to *combine* those known features into per-pixel class predictions. This is called **transfer learning** and it's the standard move when you have very little data.

### What "21 channels" means in the output
The network's last layer outputs a tensor of shape `21 × H × W`. For each pixel it produces 21 numbers — a score for each Pascal VOC class. We take `argmax` over the 21 scores at every pixel to get the predicted class ID, which becomes the segmentation mask.

### Loss (`CrossEntropyLoss(ignore_index=255)`)
Cross-entropy is the standard loss for "pick the right class out of N options" problems. Applied per pixel here. The `ignore_index=255` part tells PyTorch to skip pixels labeled 255 in the mask (those are object boundaries marked as "don't care") — they don't contribute to the loss or its gradient.

### Architecture / training details

- **Encoder:** `torchvision.models.resnet18(weights=IMAGENET1K_V1)`, take feature maps after:
  - conv1+bn+relu (64ch, /2)
  - layer1 (64ch, /4)
  - layer2 (128ch, /8)
  - layer3 (256ch, /16)
  - layer4 (512ch, /32)  ← bottleneck
- **Decoder:** 4 up-blocks (bilinear upsample + concat skip + 2×conv-bn-relu). Final 1×1 conv → 21 classes.
- **Output:** 21 channels (Pascal VOC classes 0..20).
- **Loss:** `CrossEntropyLoss(ignore_index=255)`.
- **Optimizer:** Adam, lr=1e-3 (encoder gets lr=1e-4 — lower, since pretrained).
- **Input size:** resize to 256×256 (bilinear for images, nearest for masks). ImageNet mean/std normalization on input.
- **Augmentation (train only):** random horizontal flip, random crop after resize-to-288, mild color jitter.
- **Epochs:** ~80 (less needed with pretraining). Loss curve saved to `train_loss.png`.
- **Device:** CUDA if available, else CPU.

### Inference / evaluation
After training we freeze the weights and just run the network on test images.
- Load each test image, resize to 256×256, run model, argmax → predicted class map.
- Resize prediction back to original size with **nearest-neighbor** (so we don't accidentally blend class IDs into nonsense in-between values).
- Save three views per image:
  - **Predicted mask** colored with the VOC palette so you can see the classes.
  - **Overlay** — the colored mask blended on top of the original photo at ~50% alpha.
  - **Side-by-side** — original | ground-truth mask | prediction, for visual comparison.

### Metrics
Two standard segmentation metrics, both ignoring pixels labeled 255:
- **Pixel accuracy** = (# pixels predicted correctly) / (total non-ignore pixels). Easy to read, but inflated by the dominant "background" class.
- **Mean IoU (Intersection-over-Union)** = average of per-class `intersection / union`. Penalizes both false positives and missed pixels and is the standard segmentation metric. A class with no ground-truth pixels in an image is skipped for that image.

### Honest caveats (will print, also in metrics.txt)
- 20 training images is far too few — model will overfit and generalize poorly.
- Test set contains classes (8 cat, 10 cow, 19 train) never seen in training → those pixels can't be predicted correctly. Expected.
- Goal of the assignment is comparison/methodology, not SOTA.

## main.py structure
```
def voc_palette() -> np.ndarray
def load_pair(jpg_path)              # returns image (RGB), mask (HxW indices) or None
def kmeans_numpy(X, K, seed, ...)    # custom impl
def run_task1(data_dir, out_dir)
class UNet(nn.Module)                # tiny U-Net
class SegDataset(Dataset)
def train_unet(...)
def evaluate_unet(...)
def run_task2(train_dir, test_dir, out_dir)
if __name__ == "__main__":
    run_task1(...)
    run_task2(...)
```

## Report (`report.md`) structure
Written after `main.py` finishes so it can reference actual numbers and figures.

1. **Title / header** — course, assignment, name.
2. **Dataset** — 1 paragraph: Pascal VOC subset, 20 train + 10 test pairs, mask format (indexed PNG, 21 classes + ignore=255), note which classes appear in train vs test.
3. **Task 1 — K-means**
   - Method (K=4, RGB, custom NumPy implementation, two seeds).
   - 2–3 example figures (embed `comparison.png` from representative images — e.g., a plane, a person, a multi-object scene).
   - Analysis: how initialization changed the output, what K-means captured well (color regions, sky/ground split), what it missed (semantic identity, thin structures, shadow boundaries).
4. **Task 2 — U-Net with pretrained ResNet-18**
   - Architecture diagram/description (encoder → bottleneck → decoder with skips, 21-class output).
   - Training setup (optimizer, lr, epochs, augmentation, loss, ignore index).
   - Training loss curve (embed `train_loss.png`).
   - Qualitative results: 2–3 `<image_id>_compare.png` figures.
   - Quantitative results: table of per-class IoU + mean IoU + pixel accuracy (from `metrics.txt`).
   - Strengths vs K-means (semantic understanding, cleaner boundaries on known classes).
   - Weaknesses (tiny training set, failure on unseen classes like cat/cow/train, overfitting signal from the loss curve).
5. **Comparison / conclusion** — short paragraph: when would you reach for K-means vs a learned model, what the assignment demonstrates.

## Open questions / assumptions
- "Training" set vs "testing" set: I'll use the 20 pairs in `segmentation_data/` for training and `testing-dataset/` for testing (matches folder layout).
- K-means runs on **all 30** images (training + testing) so Task 1 is independent of any train/test split.
- Pretrained ResNet-18 weights will be downloaded from torchvision on first run (~45 MB). Requires internet once; cached afterwards.
