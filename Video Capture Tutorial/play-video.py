import cv2
import argparse


def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print("Press 'q' to exit playback.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends

        cv2.imshow("Video Player", frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play a video file.")
    parser.add_argument("video", type=str, help="Path to the video file")
    args = parser.parse_args()

    play_video(args.video)


if __name__ == "__main__":
    main()
