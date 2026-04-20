# Term Project — Implementation Plan

This document walks through how we'll build the scene recognition pipeline step by step. If you're new to computer vision, each section includes a short explanation of **what** the technique is and **why** we use it before getting into the how.

---

## Big Picture

We have photos of 4 types of scenes (e.g. forest, bedroom, kitchen, coast). Our job is to train classifiers that can look at a **new** photo and say "this is a forest." We'll try four different approaches — from dead-simple to deep learning — and compare how well each one works.

```
Train Images ──► Feature Extraction ──► Classifier ──► Predict labels on Test Images ──► Report accuracy
```

---

## Project Structure

```
Term-Project/
├── Data_S26/
│   ├── Train/          # 600 training images (150 per category)
│   │   ├── bedroom/
│   │   ├── desert/
│   │   ├── landscape/
│   │   └── rainforest/
│   └── Test/           # 200 test images (50 per category)
│       ├── bedroom/
│       ├── desert/
│       ├── landscape/
│       └── rainforest/
├── output/
│   ├── gray_200/       # pre-processed grayscale 200x200
│   ├── gray_50/        # pre-processed grayscale 50x50
│   ├── sift/           # saved SIFT feature files (.npy)
│   └── hist/           # saved histogram feature files (.npy)
├── main.py             # single entry-point script that runs everything
├── project.md          # assignment spec
├── plan.md             # this file
└── report.pdf          # final write-up
```

We'll keep it all in **one script** (`main.py`) broken into clearly labeled sections so it's easy to follow and easy to submit.

---

## Step 0 — Setup & Data Loading

**What we need:** Python 3, plus the libraries already in `requirements.txt`:

- `opencv-contrib-python` — image I/O, resizing, SIFT, histograms
- `numpy` — array math
- `torch` / `torchvision` — building and training the CNN
- `matplotlib` — optional, for visualizing results

**How it works:**

1. Walk through `Data_S26/Train/<category>/` and `Data_S26/Test/<category>/` folders.
2. The sub-folder name **is** the label — `bedroom`, `desert`, `landscape`, `rainforest`.
3. Build a list of `(file_path, label)` pairs for train (600 images) and test (200 images).

> **Jargon buster — "label":** Just a tag that says what category the image belongs to. The computer doesn't inherently know what a "forest" is; we tell it by putting forest photos in a folder called `forest`.

---

## Step 1 — Pre-processing

### 1a. Grayscale + Brightness Adjustment

**Why grayscale?** Color adds complexity (3 channels instead of 1) but doesn't help much for scene _structure_. Converting to grayscale keeps things simpler and faster.

**Why adjust brightness?** If an image is too dark or too bright, features get washed out. We normalize brightness so the classifiers see consistent input.

**How:**

```python
import cv2
import numpy as np

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Average brightness is the mean pixel value scaled to [0, 1]
avg_brightness = np.mean(gray) / 255.0

if avg_brightness < 0.4:
    # Increase brightness — multiply pixel values by a factor > 1
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
elif avg_brightness > 0.6:
    # Decrease brightness
    gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-30)
```

- `alpha` scales contrast, `beta` shifts brightness. Think of it like `new_pixel = alpha * old_pixel + beta`.

### 1b. Resize to 200x200 and 50x50

```python
img_200 = cv2.resize(gray, (200, 200))
img_50  = cv2.resize(gray, (50, 50))
```

Save both versions to `output/gray_200/` and `output/gray_50/` (keeping category sub-folders so we can load them later with labels intact).

---

## Step 2 — SIFT Feature Extraction

### What is SIFT?

**SIFT (Scale-Invariant Feature Transform)** finds interesting "keypoints" in an image — corners, blobs, edges — and describes each one with a 128-number vector (called a **descriptor**). These descriptors don't change much if the image is rotated, scaled, or slightly shifted, which makes them great for recognition.

Think of it like this: SIFT looks at a photo of a forest and finds dozens of little patches (a tree trunk edge, a leaf cluster, a shadow boundary) and creates a fingerprint for each patch.

### The problem: variable number of keypoints

Different images produce different numbers of keypoints (one image might have 200, another might have 500). But classifiers need a **fixed-size** input. We handle this by computing a **Bag of Visual Words (BoVW)**:

1. Run SIFT on every training image → collect ALL descriptors into one big pile.
2. Cluster them into **k** groups using **K-Means** (e.g. k=50). Each cluster center is a "visual word."
3. For each image, count how many of its descriptors fall into each cluster → you get a histogram of length k. **That** is the fixed-size feature vector.

> **Jargon buster — K-Means:** An algorithm that groups similar items into k buckets. Imagine dumping 10,000 SIFT descriptors on a table and sorting them into 50 piles of similar-looking patches.

**How:**

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_200, None)
# descriptors is an (N, 128) array — N keypoints, each described by 128 numbers
```

Then use `sklearn.cluster.KMeans` or OpenCV's `cv2.BOWKMeansTrainer` to build the vocabulary and `cv2.BOWImgDescriptorExtractor` to compute per-image histograms. Alternatively we can do this manually with NumPy — it's straightforward.

Save the resulting feature vectors as `.npy` files.

---

## Step 3 — Histogram Feature Extraction

### What is a histogram feature?

A **grayscale histogram** counts how many pixels in the image have each intensity value (0–255). It's a simple 256-number summary of the image's overall tone.

- A dark forest scene will have lots of low values (peaks on the left).
- A bright beach scene will have peaks on the right.

This is a very coarse feature — it ignores _where_ pixels are — but it's fast and sometimes surprisingly effective for distinguishing scene types with very different lighting profiles.

**How:**

```python
hist = cv2.calcHist([gray_200], [0], None, [256], [0, 256])
hist = hist.flatten()
hist = hist / hist.sum()  # normalize so totals = 1
```

Save as `.npy` files.

---

## Step 4 — Training the Four Classifiers

### 4a. Nearest Neighbor on Raw 50x50 Pixels

**What is Nearest Neighbor (k-NN)?**

The simplest classifier there is. To classify a test image:

1. Flatten it into a vector of 2500 numbers (50 \* 50).
2. Compare it to **every** training image vector using Euclidean distance.
3. The training image that is closest → its label becomes the prediction.

No actual "training" happens — we just store the training data and compare at test time. That's why it's called a **lazy learner**.

**How:**

```python
# Training: just flatten and stack
X_train = np.array([img.flatten() for img in train_50_images])  # shape (N, 2500)
y_train = np.array(train_labels)

# Predicting: for a test image
test_vec = test_img.flatten()
distances = np.linalg.norm(X_train - test_vec, axis=1)
predicted_label = y_train[np.argmin(distances)]
```

We can use `sklearn.neighbors.KNeighborsClassifier` for convenience or write it ourselves — it's only a few lines.

### 4b. Nearest Neighbor on SIFT features

Same k-NN approach, but instead of raw pixels we use the BoVW histogram vectors from Step 2.

### 4c. Nearest Neighbor on Histogram features

Same k-NN approach, using the 256-d histogram vectors from Step 3.

### 4d. CNN on 200x200 images

**What is a CNN?**

A **Convolutional Neural Network** learns its own features automatically by sliding small filters across the image. Early layers learn simple patterns (edges, textures); deeper layers combine those into complex patterns (tree canopies, bed frames). A final fully-connected layer maps those learned features to class scores.

**Architecture (keep it simple):**

```
Input (1, 200, 200)           # 1 channel (grayscale), 200x200
  → Conv2d(1, 16, 3, padding=1) + ReLU + MaxPool(2)    # → (16, 100, 100)
  → Conv2d(16, 32, 3, padding=1) + ReLU + MaxPool(2)   # → (32, 50, 50)
  → Conv2d(32, 64, 3, padding=1) + ReLU + MaxPool(2)   # → (64, 25, 25)
  → Flatten                                              # → 64*25*25 = 40000
  → Linear(40000, 128) + ReLU
  → Linear(128, 4)                                       # 4 scene categories
```

**Training loop:**

- Loss function: `CrossEntropyLoss` (standard for classification).
- Optimizer: `Adam` with a small learning rate like `1e-3`.
- Epochs: start with 20–30, see if accuracy plateaus.
- Batch size: 16 or 32 (images are small, so this is fine on CPU).

**How (PyTorch):**

```python
import torch
import torch.nn as nn

class SceneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 25 * 25, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

> **Tip:** Use a `DataLoader` with the 200x200 grayscale images. Normalize pixel values to [0, 1] by dividing by 255.

---

## Step 5 — Testing & Evaluation

Run each of the 4 classifiers on **every** test image and compute:

| Metric                  | What it means                                                          |
| ----------------------- | ---------------------------------------------------------------------- |
| **Accuracy**            | % of test images whose predicted label matches the true label          |
| **False Positive rate** | % of test images that were assigned to a category they don't belong to |
| **False Negative rate** | % of test images that were not assigned to their correct category      |

For a multi-class problem, FP and FN are computed **per class** and then averaged. Use a **confusion matrix** to make this easy:

```python
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

This gives precision, recall, and F1 per class — FP and FN rates fall right out of the confusion matrix.

---

## Step 6 — Write-Up

The report should cover:

1. **Implementation** — briefly describe each step (a few sentences each, not full code).
2. **Results table** — accuracy / FP / FN for all 4 methods side by side.
3. **Analysis** — expected findings:
   - Raw pixels (4a) will likely perform worst — it's comparing raw brightness patterns with no understanding of structure.
   - SIFT (4b) should do better — it captures local structure regardless of exact position.
   - Histograms (4c) might surprise you — cheap but effective if the 4 scene types have different brightness distributions.
   - CNN (4d) should perform best — it learns the right features for the task.
4. **Discussion** — what affected accuracy? Training set size, image quality, hyperparameters, etc.

---

## Implementation Order (Checklist)

- [ ] Set up folder structure and data loading
- [ ] Pre-processing: grayscale + brightness + resize (Step 1)
- [ ] Histogram features (Step 3) — easiest feature to extract
- [ ] k-NN on raw pixels (Step 4a) — simplest classifier to get working
- [ ] k-NN on histograms (Step 4c) — just swap the feature vector
- [ ] SIFT + BoVW (Step 2) — more involved, do after simpler methods work
- [ ] k-NN on SIFT (Step 4b)
- [ ] CNN (Step 4d) — most code, do last
- [ ] Evaluation on test set (Step 5)
- [ ] Write report (Step 6)

> **Why this order?** We start with the simplest pieces so we can verify the data pipeline works before adding complexity. If raw-pixel k-NN runs and gives _some_ accuracy, we know the loading/labeling is correct and can confidently build on top of it.
