import cv2
import argparse
import numpy as np
from ffpyplayer.player import MediaPlayer


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)  # Load audio

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print("Press 'q' to exit playback.")

    while True:
        ret, frame = cap.read()
        audio_frame, val = player.get_frame()  # Get audio frame

        if not ret:
            break  # End of video

        cv2.imshow("Video Player", frame)

        # Play audio if available
        if val != "eof" and audio_frame is not None:
            img, t = audio_frame  # Extract audio frame

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Play a video file with audio.")
    parser.add_argument("video", type=str, help="Path to the video file")
    args = parser.parse_args()

    play_video(args.video)


if __name__ == "__main__":
    main()
