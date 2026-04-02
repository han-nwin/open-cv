# Assignment 3 - Plan

## CNN Crash Course

### What is a CNN?
A Convolutional Neural Network (CNN) is a type of neural network designed to work with images.
Regular neural networks take a flat list of numbers as input, but images have spatial structure -
pixels next to each other matter. CNNs preserve that spatial relationship.

Note: We convert all images to **grayscale** first, so each image is a single 2D grid of pixel
values (0-255). A color image would have 3 channels (R, G, B), but grayscale = 1 channel.
This keeps things simpler and is enough for handwriting recognition since we only care about
the ink strokes, not color.

### What we're trying to achieve
We want the model to look at a handwriting image and answer one question:
**"Is this Han's handwriting or someone else's?"** - binary classification (own = 1, others = 0).

### Why these layers?

**Convolutional Layer (Conv2d)** - The core idea. A small filter (3x3 grid of weights) slides across
the image and detects patterns like edges, curves, and strokes. The first conv layer detects simple
features (lines, edges). The second conv layer combines those into more complex features
(loops, corners, letter shapes). Think of it like the network learning "what does Han's handwriting look like?"

A channel is a whole grid. The input has 1 channel (grayscale). Each conv layer
produces multiple channels, where each channel is a "feature map" - a full grid showing how
strongly one specific pattern (edges, curves, etc.) was detected at each location. The network
learns what patterns to look for during training.

**How many channels?** There's no strict rule, but a common convention is to double each layer:
```
32 -> 64:    fewer parameters, less likely to overfit, might miss subtle patterns
64 -> 128:   more parameters, captures more detail, needs more data to train well
128 -> 256:  a lot of parameters, probably overkill for a small dataset
```
We go with **32 -> 64** since our dataset is small and this is less likely to overfit.

```
Input: 1 channel (grayscale image)
+------------------+
|  128 x 128       |  <- raw pixel values
|  (the image)     |
+------------------+

After Conv1 + ReLU + Pool: 32 channels, each 64x64
+--------+ +--------+ +--------+     +--------+
| 64x64  | | 64x64  | | 64x64  | ... | 64x64  |
| edges  | | curves | | lines  |     | ???    |
+--------+ +--------+ +--------+     +--------+
  ch 1       ch 2       ch 3           ch 32
  (each channel is a different learned feature detector)

After Conv2 + ReLU + Pool: 64 channels, each 32x32
+------+ +------+ +------+     +------+
|32x32 | |32x32 | |32x32 | ... |32x32 |
| loops| |corners| |shapes|    | ???  |
+------+ +------+ +------+     +------+
  ch 1     ch 2     ch 3         ch 64
  (combines simple features into complex patterns)
```

**How convolution actually works:**

```
One filter sliding across the image:

Image (128x128):          Filter (3x3):        Output (128x128):
+---+---+---+---+--       +---+---+---+        +---+---+---+---+--
| 10| 20| 30| 40|..       | 1 | 0 |-1 |        | 50|   |   |   |..
+---+---+---+---+--       +---+---+---+   =>   +---+---+---+---+--
| 50| 60| 70| 80|..       | 1 | 0 |-1 |        |   |   |   |   |..
+---+---+---+---+--       +---+---+---+        +---+---+---+---+--
| 90|100|110|120|..       | 1 | 0 |-1 |        |   |   |   |   |..
+---+---+---+---+--                             +---+---+---+---+--

The filter sits at top-left, does the math:
(10*1 + 20*0 + 30*-1) + (50*1 + 60*0 + 70*-1) + (90*1 + 100*0 + 110*-1) = 50
Then it slides right one pixel, repeats. Covers the whole image.
```

With padding=1, the output stays 128x128 (padding adds zeros around the border so the
filter can cover edges). Each filter produces one 128x128 feature map.
32 filters = 32 feature maps of 128x128.

**MaxPool (Pooling)** - Then MaxPool(2) shrinks each 128x128 down to 64x64 by taking the
max in each 2x2 block:

```
Before pool:              After pool:
+---+---+---+---+         +---+---+
| 10| 20| 30| 40|         | 60| 80|    (max of each 2x2 block)
+---+---+---+---+   =>   +---+---+
| 50| 60| 70| 80|
+---+---+---+---+
```

So: 1 image -> 32 filters each produce 128x128 -> pool shrinks to 64x64 -> **32 channels of 64x64**.

Then conv2 does the same thing again on those 32 channels:

```
Conv2: 64 filters, each looks at ALL 32 channels at once
+--------+ +--------+     +--------+
| 64x64  | | 64x64  | ... | 64x64  |    32 channels (input)
| ch 1   | | ch 2   |     | ch 32  |
+--------+ +--------+     +--------+
     \          |           /
      \         |          /
       v        v         v
    one filter reads all 32 channels,
    combines them into 1 output grid
            |
            v
       +--------+
       | 64x64  |   1 feature map (before pool)
       +--------+
            |  x64 filters = 64 feature maps
            v
       +------+
       |32x32 |   after pool (64x64 -> 32x32)
       +------+
            |  x64 = 64 channels of 32x32
            v
     DONE with conv layers!
     Now flatten and classify.
```

**ReLU (Activation)** - After each convolution, we apply ReLU which just means: if the value is
negative, make it 0. This adds non-linearity so the network can learn complex patterns instead of
just simple linear combinations.

**MaxPool (Pooling)** - Shrinks the image by taking the max value in each 2x2 block, cutting
dimensions in half. This makes the network faster, reduces overfitting, and makes it care less
about the exact position of features (a letter "A" shifted a few pixels should still be recognized).

**Flatten** - After the conv layers, we have a 3D block of features (64 channels x 32 x 32).
Flatten turns this into a single long vector (65,536 numbers) so we can feed it into a regular
dense layer. By this point, the "seeing" is already done - the conv layers already extracted
meaningful features. Each number represents something like "how much curve was in this region."

```
Before flatten:
  Channel 1:  [32x32 grid]
  Channel 2:  [32x32 grid]
  ...
  Channel 64: [32x32 grid]

After flatten (channel 1 first, row by row, then channel 2, etc.):
  [ch1_r1_c1, ch1_r1_c2, ..., ch1_r32_c32, ch2_r1_c1, ..., ch64_r32_c32]
  = 32*32*64 = 65,536 numbers
```

The order doesn't matter to the Dense layer - it just learns which positions correspond to
which features, as long as the order stays consistent during training and prediction.

**Dense / Linear (Fully Connected)** - Takes all those extracted features and learns which
combinations of features mean "own" vs "others". This is where the actual classification decision happens.

**Sigmoid (Output)** - Squishes the output to a value between 0 and 1. Close to 1 = own
handwriting, close to 0 = others. We use 0.5 as the cutoff.

### The flow
```
Raw Image -> [Conv: find edges] -> [ReLU] -> [Pool: shrink]
          -> [Conv: find shapes] -> [ReLU] -> [Pool: shrink]
          -> [Flatten: make 1D]
          -> [Dense: decide which features matter]
          -> [Sigmoid: output probability 0-1]
```

---

## 1. Dataset Preparation
- Organize images into `dataset/train/own/`, `dataset/train/others/`, `dataset/test/own/`, `dataset/test/others/`
- 80/20 split between train and test
- Label: own = 1, others = 0

## 2. Image Preprocessing (OpenCV)
- `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` - convert to grayscale
- `cv2.resize(img, (128, 128))` - resize to fixed size
- `cv2.threshold()` or `cv2.adaptiveThreshold()` - binarize
- `cv2.GaussianBlur()` - optional noise removal
- `cv2.morphologyEx()` - optional morphological cleanup
- Normalize pixel values to 0-1 by dividing by 255.0

## 3. Data Loading (PyTorch)
- Custom `Dataset` class that:
  - Reads images from `dataset/train/` and `dataset/test/`
  - Applies preprocessing in `__getitem__`
  - Returns `(tensor, label)` where tensor is shape `(1, 128, 128)`
- `DataLoader` with batch size ~16-32, shuffle=True for training

## 4. CNN Model (PyTorch)
```python
import torch.nn as nn

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
```

## 5. Training
- Loss: `nn.BCELoss()`
- Optimizer: `torch.optim.Adam(model.parameters(), lr=0.001)`
- Loop for ~20-30 epochs:
  - Forward pass, compute loss
  - Backward pass, optimizer step
  - Print training accuracy each epoch
- Save model with `torch.save(model.state_dict(), 'model.pth')`

## 6. Evaluation
- Run model on test DataLoader with `torch.no_grad()`
- Compute:
  - Accuracy = correct / total
  - False Positive = predicted own but actually others
  - True Negative = correctly predicted others
- Print confusion matrix style results

## 7. Visualization & Prediction
- `cv2.imread()` a test image
- Preprocess it the same way
- Run through model, get prediction
- `cv2.putText()` the label on the image
- `cv2.imshow()` to display

## 8. Write Up Report
- Dataset description & collection process
- Preprocessing choices (what worked / didn't)
- Model architecture & reasoning
- Results & failure cases
- Reflection & limitations
