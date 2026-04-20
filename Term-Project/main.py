import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Step 0 — Setup & Data Loading
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data_S26")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CATEGORIES = ["bedroom", "desert", "landscape", "rainforest"]


def load_image_paths(split):
    """Load (file_path, label) pairs for a given split ('Train' or 'Test')."""
    paths = []
    for cat in CATEGORIES:
        folder = os.path.join(DATA_DIR, split, cat)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                paths.append((fpath, cat))
    return paths


# ============================================================
# Step 1 — Pre-processing: Grayscale + Brightness + Resize
# ============================================================


def preprocess(image_path):
    """Convert to grayscale, adjust brightness, resize to 200x200 and 50x50."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Brightness adjustment
    avg_brightness = np.mean(gray) / 255.0
    if avg_brightness < 0.4:
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    elif avg_brightness > 0.6:
        gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-30)

    img_200 = cv2.resize(gray, (200, 200))
    img_50 = cv2.resize(gray, (50, 50))
    return img_200, img_50


def run_preprocessing():
    """Pre-process all training images and save to output/gray_200 and output/gray_50."""
    train_paths = load_image_paths("Train")

    for size_label in ["gray_200", "gray_50"]:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(OUTPUT_DIR, size_label, cat), exist_ok=True)

    print(f"Pre-processing {len(train_paths)} training images...")
    for i, (fpath, cat) in enumerate(train_paths):
        fname = os.path.basename(fpath)
        img_200, img_50 = preprocess(fpath)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "gray_200", cat, fname), img_200)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "gray_50", cat, fname), img_50)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(train_paths)} done")

    print("Pre-processing complete!")
    print(f"  200x200 images saved to: output/gray_200/")
    print(f"  50x50 images saved to:   output/gray_50/")


# ============================================================
# Step 2 — SIFT Feature Extraction (Bag of Visual Words)
# ============================================================

SIFT_K = 50  # number of visual words (clusters)


def extract_all_sift_descriptors():
    """Run SIFT on all 200x200 training images, return list of (descriptors, label)."""
    sift = cv2.SIFT_create()
    all_descriptors = []  # one big list for K-Means
    per_image = []  # (descriptors, label) per image

    for cat in CATEGORIES:
        folder = os.path.join(OUTPUT_DIR, "gray_200", cat)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(img, None)
            if des is not None:
                all_descriptors.append(des)
                per_image.append((des, cat))
            else:
                # No keypoints found — store empty
                per_image.append((np.zeros((0, 128), dtype=np.float32), cat))

    return all_descriptors, per_image


def build_bovw_vocabulary(all_descriptors, k=SIFT_K):
    """Cluster all SIFT descriptors into k visual words using K-Means."""
    stacked = np.vstack(all_descriptors).astype(np.float32)
    print(f"  Clustering {stacked.shape[0]} descriptors into {k} visual words...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(
        stacked, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    return centers


def descriptors_to_bovw(descriptors, centers):
    """Convert a single image's SIFT descriptors to a BoVW histogram."""
    k = centers.shape[0]
    if descriptors.shape[0] == 0:
        return np.zeros(k, dtype=np.float32)

    # Assign each descriptor to nearest cluster center
    # distances shape: (num_descriptors, k)
    diffs = descriptors[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    assignments = np.argmin(dists, axis=1)

    hist = np.bincount(assignments, minlength=k).astype(np.float32)
    hist /= hist.sum() + 1e-7  # normalize
    return hist


def run_sift_extraction():
    """Extract SIFT features, build vocabulary, compute BoVW histograms, save."""
    print("Extracting SIFT descriptors from training images...")
    all_descriptors, per_image = extract_all_sift_descriptors()
    print(f"  Found descriptors in {len(all_descriptors)} images")

    centers = build_bovw_vocabulary(all_descriptors)

    # Save vocabulary for use at test time
    os.makedirs(os.path.join(OUTPUT_DIR, "sift"), exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "sift", "vocabulary.npy"), centers)

    # Compute BoVW histogram for each training image
    features = []
    labels = []
    for des, cat in per_image:
        hist = descriptors_to_bovw(des, centers)
        features.append(hist)
        labels.append(cat)

    features = np.array(features)
    labels = np.array(labels)
    np.save(os.path.join(OUTPUT_DIR, "sift", "train_features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "sift", "train_labels.npy"), labels)

    print(f"  BoVW features shape: {features.shape}")
    print(f"  Saved to: output/sift/")


# ============================================================
# Step 3 — Histogram Feature Extraction
# ============================================================


def run_histogram_extraction():
    """Extract grayscale histogram features from all training images, save."""
    os.makedirs(os.path.join(OUTPUT_DIR, "hist"), exist_ok=True)

    features = []
    labels = []

    for cat in CATEGORIES:
        folder = os.path.join(OUTPUT_DIR, "gray_200", cat)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-7)  # normalize
            features.append(hist)
            labels.append(cat)

    features = np.array(features)
    labels = np.array(labels)
    np.save(os.path.join(OUTPUT_DIR, "hist", "train_features.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "hist", "train_labels.npy"), labels)

    print(f"Histogram features extracted!")
    print(f"  Feature shape: {features.shape}")
    print(f"  Saved to: output/hist/")


# ============================================================
# Step 4 — Training & Classification
# ============================================================

# --- 4a, 4b, 4c: Nearest Neighbor ---


def knn_predict(X_train, y_train, X_test):
    """1-Nearest Neighbor: for each test sample, find closest training sample."""
    predictions = []
    for test_vec in X_test:
        dists = np.linalg.norm(X_train - test_vec, axis=1)
        predictions.append(y_train[np.argmin(dists)])
    return np.array(predictions)


def prepare_test_features_pixels():
    """Load test images as 50x50, preprocess same as training, return (features, labels)."""
    test_paths = load_image_paths("Test")
    features = []
    labels = []
    for fpath, cat in test_paths:
        _, img_50 = preprocess(fpath)
        features.append(img_50.flatten().astype(np.float32))
        labels.append(cat)
    return np.array(features), np.array(labels)


def prepare_train_features_pixels():
    """Load all 50x50 training images as flat vectors."""
    features = []
    labels = []
    for cat in CATEGORIES:
        folder = os.path.join(OUTPUT_DIR, "gray_50", cat)
        for fname in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            features.append(img.flatten().astype(np.float32))
            labels.append(cat)
    return np.array(features), np.array(labels)


def prepare_test_features_sift():
    """Extract SIFT BoVW features for test images using saved vocabulary."""
    sift = cv2.SIFT_create()
    centers = np.load(os.path.join(OUTPUT_DIR, "sift", "vocabulary.npy"))
    test_paths = load_image_paths("Test")
    features = []
    labels = []
    for fpath, cat in test_paths:
        img_200, _ = preprocess(fpath)
        _, des = sift.detectAndCompute(img_200, None)
        if des is None:
            des = np.zeros((0, 128), dtype=np.float32)
        hist = descriptors_to_bovw(des, centers)
        features.append(hist)
        labels.append(cat)
    return np.array(features), np.array(labels)


def prepare_test_features_hist():
    """Extract histogram features for test images."""
    test_paths = load_image_paths("Test")
    features = []
    labels = []
    for fpath, cat in test_paths:
        img_200, _ = preprocess(fpath)
        hist = cv2.calcHist([img_200], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.append(hist)
        labels.append(cat)
    return np.array(features), np.array(labels)


# --- 4d: CNN ---


class SceneDataset(Dataset):
    """PyTorch dataset that loads 200x200 grayscale images."""

    def __init__(self, split):
        if split == "Train":
            self.samples = []
            for cat_idx, cat in enumerate(CATEGORIES):
                folder = os.path.join(OUTPUT_DIR, "gray_200", cat)
                for fname in sorted(os.listdir(folder)):
                    self.samples.append((os.path.join(folder, fname), cat_idx))
        else:
            self.samples = []
            test_paths = load_image_paths("Test")
            for fpath, cat in test_paths:
                self.samples.append((fpath, CATEGORIES.index(cat)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        if "gray_200" in fpath:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        else:
            img_200, _ = preprocess(fpath)
            img = img_200
        # Normalize to [0, 1], add channel dim -> (1, 200, 200)
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, label


class SceneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_cnn(epochs=30, batch_size=16, lr=1e-3):
    """Train CNN and return the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    train_dataset = SceneDataset("Train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SceneCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        if (epoch + 1) % 5 == 0:
            acc = correct / total * 100
            avg_loss = total_loss / total
            print(
                f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.1f}%"
            )

    return model, device


def evaluate_cnn(model, device):
    """Run CNN on test set, return (y_true, y_pred) as category name arrays."""
    test_dataset = SceneDataset("Test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array([CATEGORIES[i] for i in all_labels])
    y_pred = np.array([CATEGORIES[i] for i in all_preds])
    return y_true, y_pred


# ============================================================
# Step 5 — Evaluation
# ============================================================


def evaluate(y_true, y_pred, method_name):
    """Print accuracy, per-class FP/FN rates. Returns a formatted string for saving."""
    accuracy = np.mean(y_true == y_pred) * 100

    lines = []
    lines.append(f"[{method_name}]")
    lines.append(f"Accuracy: {accuracy:.1f}%")
    lines.append(f"{'Category':<14} {'FP Rate':>8} {'FN Rate':>8}")
    lines.append("-" * 32)

    for cat in CATEGORIES:
        fp = np.sum((y_pred == cat) & (y_true != cat))
        fn = np.sum((y_true == cat) & (y_pred != cat))
        total_not_cat = np.sum(y_true != cat)
        total_cat = np.sum(y_true == cat)

        fp_rate = fp / total_not_cat * 100 if total_not_cat > 0 else 0
        fn_rate = fn / total_cat * 100 if total_cat > 0 else 0
        lines.append(f"{cat:<14} {fp_rate:>7.1f}% {fn_rate:>7.1f}%")

    block = "\n".join(lines)
    print(f"\n  {block.replace(chr(10), chr(10) + '  ')}\n")
    return block


def run_classification():
    """Run all 4 classifiers on test data, report results, save to results.txt."""
    results = []

    print("=" * 50)
    print("STEP 4 & 5 — Classification & Evaluation")
    print("=" * 50)

    # 4a: k-NN on raw 50x50 pixels
    print("\n[4a] k-NN on raw 50x50 pixels...")
    X_train_px, y_train_px = prepare_train_features_pixels()
    X_test_px, y_test_px = prepare_test_features_pixels()
    y_pred_px = knn_predict(X_train_px, y_train_px, X_test_px)
    results.append(evaluate(y_test_px, y_pred_px, "4a: k-NN Raw Pixels"))

    # 4b: k-NN on SIFT BoVW
    print("[4b] k-NN on SIFT BoVW features...")
    X_train_sift = np.load(os.path.join(OUTPUT_DIR, "sift", "train_features.npy"))
    y_train_sift = np.load(os.path.join(OUTPUT_DIR, "sift", "train_labels.npy"))
    X_test_sift, y_test_sift = prepare_test_features_sift()
    y_pred_sift = knn_predict(X_train_sift, y_train_sift, X_test_sift)
    results.append(evaluate(y_test_sift, y_pred_sift, "4b: k-NN SIFT BoVW"))

    # 4c: k-NN on Histogram
    print("[4c] k-NN on Histogram features...")
    X_train_hist = np.load(os.path.join(OUTPUT_DIR, "hist", "train_features.npy"))
    y_train_hist = np.load(os.path.join(OUTPUT_DIR, "hist", "train_labels.npy"))
    X_test_hist, y_test_hist = prepare_test_features_hist()
    y_pred_hist = knn_predict(X_train_hist, y_train_hist, X_test_hist)
    results.append(evaluate(y_test_hist, y_pred_hist, "4c: k-NN Histogram"))

    # 4d: CNN
    print("[4d] Training CNN...")
    model, device = train_cnn()
    y_true_cnn, y_pred_cnn = evaluate_cnn(model, device)
    results.append(evaluate(y_true_cnn, y_pred_cnn, "4d: CNN"))

    # Save results to file
    results_path = os.path.join(BASE_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write("Scene Recognition and Classification Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(("\n\n" + "-" * 50 + "\n\n").join(results))
        f.write("\n")
    print(f"Results saved to: {results_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_preprocessing()
    run_sift_extraction()
    run_histogram_extraction()
    run_classification()
