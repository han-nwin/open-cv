import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. Image Preprocessing (OpenCV)
# ============================================================
def preprocess_image(img_path):
    """Read an image and preprocess it for the CNN."""
    img = cv2.imread(img_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 128x128
    resized = cv2.resize(gray, (128, 128))
    # Gaussian blur for noise removal
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    # Binarize with adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Normalize pixel values to 0-1
    normalized = binary.astype(np.float32) / 255.0
    return normalized


# ============================================================
# 2. Dataset Class (PyTorch)
# ============================================================
class SignatureDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: e.g. 'dataset/train' or 'dataset/test'
        Expects subfolders: own/ and others/
        """
        self.images = []
        self.labels = []

        # Load "own" images (label = 1)
        own_dir = os.path.join(root_dir, "own")
        for fname in os.listdir(own_dir):
            fpath = os.path.join(own_dir, fname)
            if os.path.isfile(fpath):
                self.images.append(fpath)
                self.labels.append(1.0)

        # Load "others" images (label = 0)
        others_dir = os.path.join(root_dir, "others")
        for fname in os.listdir(others_dir):
            fpath = os.path.join(others_dir, fname)
            if os.path.isfile(fpath):
                self.images.append(fpath)
                self.labels.append(0.0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = preprocess_image(self.images[idx])
        # Add channel dimension: (128, 128) -> (1, 128, 128)
        img_tensor = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(self.labels[idx])
        return img_tensor, label


# ============================================================
# 3. CNN Model
# ============================================================
class SignatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        #   input: (1, 128, 128) -> output: (32, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # Layer 2: Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
        #   output: (64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # Flatten: 64 * 32 * 32 = 65536
        # Dense: Linear(65536, 128) -> ReLU
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        # Output: Linear(128, 1) -> Sigmoid
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x)))  # (1,128,128) -> (32,64,64)
        # Layer 2: Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))  # (32,64,64) -> (64,32,32)
        # Flatten: 64 * 32 * 32 = 65536
        x = x.view(x.size(0), -1)
        # Dense: Linear(65536, 128) -> ReLU
        x = self.relu(self.fc1(x))
        # Output: Linear(128, 1) -> Sigmoid
        x = self.sigmoid(self.fc2(x))
        return x


# ============================================================
# 4. Training
# ============================================================
def train_model(model, train_loader, epochs=25, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.1f}%")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


# ============================================================
# 5. Evaluation
# ============================================================
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    false_positives = 0  # predicted own but actually others
    true_negatives = 0   # correctly predicted others

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images).squeeze(1)
            predicted = (outputs >= 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # For each sample in the batch
            for p, l in zip(predicted, labels):
                if p == 1.0 and l == 0.0:
                    false_positives += 1
                if p == 0.0 and l == 0.0:
                    true_negatives += 1

    accuracy = correct / total * 100
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives: {true_negatives}")


# ============================================================
# 6. Visualization & Prediction
# ============================================================
def predict_image(model, img_path):
    """Load a single image, predict, and display with label."""
    # Preprocess
    img = preprocess_image(img_path)
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor).item()
    label = "Own" if output >= 0.5 else "Others"
    confidence = output if output >= 0.5 else 1 - output

    # Display - save to file (WSL doesn't support cv2.imshow)
    display_img = cv2.imread(img_path)
    display_img = cv2.resize(display_img, (400, 400))
    color = (0, 255, 0) if label == "Own" else (0, 0, 255)
    cv2.putText(display_img, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # Save with label in filename
    basename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = f"prediction_{basename}.png"
    cv2.imwrite(output_path, display_img)
    print(f"  {img_path} -> {label} ({confidence:.2f}) [saved to {output_path}]")


# ============================================================
# Main
# ============================================================
def run_predictions(model):
    """Predict on all images in dataset/predict/."""
    predict_dir = "dataset/predict"
    predict_files = [f for f in os.listdir(predict_dir) if os.path.isfile(os.path.join(predict_dir, f))]
    if predict_files:
        print(f"\nPredictions on {len(predict_files)} new images:")
        for fname in sorted(predict_files):
            predict_image(model, os.path.join(predict_dir, fname))
    else:
        print("\nNo images in dataset/predict/ to predict on.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        # Load datasets
        train_dataset = SignatureDataset("dataset/train")
        test_dataset = SignatureDataset("dataset/test")

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")

        # Create model
        model = SignatureCNN()

        # Train
        train_model(model, train_loader)

        # Evaluate
        evaluate_model(model, test_loader)

        # Predict
        run_predictions(model)

    elif mode == "predict":
        # Load saved model
        model = SignatureCNN()
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        print("Loaded model from model.pth")

        run_predictions(model)
