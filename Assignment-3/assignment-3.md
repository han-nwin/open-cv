# CS 4391 Spring 2026

## Assignment 3 - Signature Recognition

**Due Date:** April 19th, 2026 11:59 PM

---

In this assignment, you will build a signature recognition system using OpenCV for preprocessing and a Convolutional Neural Networks (CNN) for classification.

**You will:**

- Collect your own handwritten signature samples (to be consistent with the training sample, please write only letters A to J in upper case as the signature sample. Try to create at least 20+ samples with various illuminations, scales, image qualities, paper background color, pen color, etc)
- Use a provided dataset of other signatures for training (80%) and testing (20%)
- Train a CNN model to distinguish your own handwriting from others
- Evaluate and visualize the system

Your final dataset should be structured in the following structure:

```
dataset/
  train/
    own/
    others/
  test/
    own/
    others/
```

You are allowed to use calls to OpenCV pre-defined functions in this assignment.

---

## Steps

### a. Image Preprocessing

1. Convert to grayscale
2. Resize to fixed size (e.g., 128x128)
3. Normalize pixel values
4. Apply thresholding or binarization
5. *Optional (Enhancements):* noise removal (blur, morphology), etc

### b. CNN Model Implementation

Build a CNN with:

1. At least 2 convolutional layers
2. Pooling layers
3. Fully connected layer
4. Output layer (binary classification)

**Example Architecture:**

```
Conv -> ReLU -> Pool
Conv -> ReLU -> Pool
Flatten -> Dense -> Output
```

### c. Training and Evaluation

#### i. Training

- Split dataset into training/testing
- Train for multiple epochs
- Track training accuracy

#### ii. Evaluation Metrics

- Accuracy
- False Positive / True Negative

### d. Visualization and Prediction

1. Load any test image in OpenCV
2. Display prediction: Own handwriting vs Others

### e. Write Up - Report

1. **Dataset Description** - How did you collect your signatures, challenges
2. **Preprocessing Choices** - What worked / what didn't
3. **Model Design** - Architecture and reasoning
4. **Results** - Accuracy, failure cases
5. **Reflection** - What would you improve? Limitations of your system

---

## Submission Instructions

Please submit your source code, dataset, and report to eLearning ONLY. (Do NOT submit your project or other files)
