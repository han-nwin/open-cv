# Assignment 3 - Signature Recognition Report

## i. Dataset Description

### Collection Process
I wrote uppercase letters A through J (ABCDEFGHIJ) by hand and took 20 photos with my iPhone. I tried to get as much variety as I could:

- Paper/background colors: white, pink, green, beige/tan, and even a blue digital screen.
- Pen/ink colors: black, red/pink, blue, yellow/gold, green, purple, and white.
- Lighting: natural light, indoor lighting, dim/dark environments, harsh yellow lighting.
- Writing sizes: some small and tight, others large and spread out.
- Surface types: plain paper, colored paper, textured surfaces, digital tablet screen.

Some images have really low contrast like white ink on beige paper or yellow on green, just to see how well the preprocessing handles it.

All images were converted from HEIC (iPhone format) to PNG since OpenCV can't read HEIC.

### Dataset Structure
- Own samples: 20 images (16 train, 4 test)
- Others samples: 61 images provided by professor (49 train, 12 test)
- 6 additional images (3 own, 3 others) set aside in `dataset/predict/` for live prediction

#### Challenges
1. At first I took pictures of the letters with spacing between letters. The model was getting like 93% accuracy which sounds good but it was basically just learning "big spacing = own" instead of actually recognizing my handwriting. So I had to retake all 20 photos on plain paper with the letters closer together to match how the "others" dataset looks.

2. HEIC format from iPhone is not supported by OpenCV so I had to convert everything to PNG.

3. Getting variety in lighting, paper color, and pen color was also a challenge since I had to borrow and buy different colored pens and paper to make the samples diverse enough.

## ii. Preprocessing Choices

### Preprocessing steps
The steps are: grayscale, resize, blur, then binarize (threshold), then normalize.

- Binarization just means turning the image into only black and white. Every pixel becomes either 0 (black) or 255 (white), nothing in between. A grayscale image has values from 0 to 255 (shades of gray), but after binarization it's only two values. This makes the ink strokes stand out clearly from the background. The source code uses `cv2.adaptiveThreshold` to do binarization.

- The order matters too. I blur before binarizing, not after. If I binarize first, I would get a noisy black and white image with random speckles. Then blurring a binary image just makes it gray again which defeats the purpose. It's like cleaning a surface before painting: smooth things out first, then apply the sharp cutoff.

- I didn't use morphology since blur + adaptive threshold was good enough.

### What worked
- Adaptive thresholding (`cv2.adaptiveThreshold`) handled the varying lighting conditions across my photos way better than simple thresholding. Since I have images with all kinds of lighting and paper colors, a fixed threshold just wasn't cutting it.
- Gaussian blur (5x5 kernel) before thresholding helped reduce noise and made the binarization a lot cleaner.
- Resizing everything to 128x128 standardized all images no matter what the original resolution was.
- Normalizing pixel values to 0 to 1 range by dividing by 255 is just standard practice for neural network input.

### What didn't
I tested three preprocessing approaches on all my images:

**v1: Fixed threshold (`cv2.threshold` with value 127), no blur.** This was the worst. On the white paper images it was ok, but on anything with color it completely failed. The blue screen image (IMG_3965) came out entirely black, you couldn't see anything. The pink paper one (IMG_3960) had the letters almost invisible. The red paper with yellow ink (IMG_3957) was just noise. A fixed threshold value can't handle different backgrounds at all.

**v2: Adaptive threshold but no blur.** Way better than v1 since it adapts to local brightness. The letters were readable on most images. But without blur there were noticeable speckles and noise in the background, especially on the dark images (IMG_3971) where the whole image was covered in noise artifacts.

**v3: Gaussian blur + adaptive threshold (what we use).** Cleanest results. The blur smooths out the noise before thresholding so the background comes out clean and the letters stay sharp. Still not perfect on the really tough images like dark backgrounds with colored ink, but it's the best of the three.

## iii. Model Design

### Architecture
```
Input (1, 128, 128) grayscale image
  -> Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)    -> (32, 64, 64)
  -> Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)   -> (64, 32, 32)
  -> Flatten                                                 -> (65536)
  -> Linear(65536, 128) -> ReLU                              -> (128)
  -> Linear(128, 1) -> Sigmoid                               -> (1) probability
```

### Reasoning
- I used 2 conv layers. The first one picks up simple features like edges and strokes, and the second one combines those into more complex patterns like letter shapes and curves.
- For the channel sizes I went with 32 then 64. The common convention is to double each layer. I kept it small since the dataset is pretty small and going bigger like 128 or 256 would probably just overfit.
- MaxPool reduces the spatial dimensions by half which makes the model faster and also makes it care less about the exact position of features. So if a letter is shifted a few pixels it should still work.
- Binary cross entropy loss with Sigmoid output is the standard setup for binary classification (own vs others). Used Adam optimizer with a learning rate of 0.001 and trained for 25 epochs. The model usually hits 100% training accuracy around epoch 14 to 17.

## iv. Results

### Training

| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 64.6% | 2.5409 |
| 5 | 75.4% | 0.5462 |
| 10 | 87.7% | 0.2197 |
| 14 | 100.0% | 0.0563 |
| 20 | 100.0% | 0.0146 |
| 25 | 100.0% | 0.0027 |

Reached 100% training accuracy by epoch 14. Loss went down to about 0.003 by epoch 25.

### Test Results

| Metric | Result |
|--------|--------|
| Accuracy | 87.5% (14/16 correct) |
| False Positives | 0 (never said someone else's writing was mine) |
| True Negatives | 12/12 (correctly rejected all "others" samples) |
| False Negatives | 2 (thought my writing was someone else's) |

### Prediction on New Images (not in train/test)

| Image | Prediction | Confidence |
|-------|-----------|------------|
| own1.png | Own | 0.89 |
| own2.png | Own | 0.99 |
| own3.png | Own | 0.76 |
| others1.png | Others | 0.99 |
| others2.png | Others | 0.97 |
| others3.png | Others | 1.00 |

All 6 predictions correct.

### Failure Cases
2 of 4 "own" test images got misclassified as "others" (false negatives). Also own3.png in predictions had lower confidence at 0.76, probably because the writing conditions for that one were different. Overall the model is more conservative, it's better at rejecting others (0 false positives) than accepting my own writing.

## v. Reflection

### What would I improve?
- More data would help a lot. Only 20 "own" samples is pretty small. More samples with more variety would make the model generalize better.
- Data augmentation could also help, like adding rotations, slight scaling, and brightness adjustments to artificially expand the dataset.
- The dataset is imbalanced with about 3x more "others" than "own" which might bias the model toward predicting "others" more often. That could explain the false negatives.
- Adding dropout layers could help reduce overfitting since there's a gap between 100% train accuracy and 87.5% test accuracy.

### Limitations
- The model overfits to training data (100% train vs 87.5% test accuracy).
- Small dataset means a few different test images could change the accuracy a lot. Each test image is worth about 6%.
- Only tested with uppercase block letters A to J so this wouldn't work for actual cursive signatures or other handwriting styles.
- Preprocessing assumes decent photo quality so really blurry or extremely low contrast images might not binarize well.
