import cv2
import numpy as np


def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_sepia(image):
    sepia_filter = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    return cv2.transform(image, sepia_filter)


def apply_negative(image):
    return cv2.bitwise_not(image)


def apply_canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)


def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)


def main():
    image_path = input("Enter the path of the image: ")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load the image. Check the file path.")
        return

    image_copy = image.copy()

    while True:
        cv2.imshow("Filtered Image", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("g"):  # Press 'g' for grayscale
            image = apply_grayscale(image_copy)
        elif key == ord("s"):  # Press 's' for sepia
            image = apply_sepia(image_copy)
        elif key == ord("n"):  # Press 'n' for negative
            image = apply_negative(image_copy)
        elif key == ord("e"):  # Press 'e' for edge detection
            image = apply_canny_edge(image_copy)
        elif key == ord("b"):  # Press 'b' for blur
            image = apply_blur(image_copy)
        elif key == ord("q"):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
