import cv2
import argparse
import numpy as np
import os


def help_message():
    print(
        "------------------------------------------------------------------------------\n"
        "This program shows how to write video files.\n"
        "You can extract the R, G, or B color channel of the input video.\n"
        "Usage:\n"
        "python video_write.py <input_video_name> [ R | G | B ] [ Y | N ]\n"
        "------------------------------------------------------------------------------\n"
    )


def main():
    help_message()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Extract color channel from video and save."
    )
    parser.add_argument("input_video", type=str, help="Path to the input video")
    parser.add_argument(
        "channel", type=str, choices=["R", "G", "B"], help="Color channel to extract"
    )
    parser.add_argument(
        "ask_output_type", type=str, choices=["Y", "N"], help="Ask for output type"
    )

    args = parser.parse_args()

    source = args.input_video
    ask_output_type = args.ask_output_type == "Y"

    # Open input video
    input_video = cv2.VideoCapture(source)
    if not input_video.isOpened():
        print(f"Could not open the input video: {source}")
        return -1

    # Extract filename and form output name
    name, ext = os.path.splitext(source)
    output_name = f"{name}_{args.channel}.avi"

    # Get codec and video properties
    fourcc = int(input_video.get(cv2.CAP_PROP_FOURCC))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # Open output video
    if ask_output_type:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Ask user or default to XVID
    output_video = cv2.VideoWriter(output_name, fourcc, fps, frame_size, True)

    if not output_video.isOpened():
        print(f"Could not open the output video for writing: {output_name}")
        return -1

    print(
        f"Input frame resolution: Width={frame_width}, Height={frame_height}, Frames={int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))}"
    )

    # Select channel index
    channel_map = {"R": 2, "G": 1, "B": 0}
    channel = channel_map[args.channel]

    while True:
        ret, frame = input_video.read()
        if not ret:
            break  # End of video

        # Extract color channel
        split_channels = list(cv2.split(frame))
        for i in range(3):
            if i != channel:
                split_channels[i] = np.zeros_like(split_channels[i])
        res = cv2.merge(split_channels)

        # Write frame to output video
        output_video.write(res)

    input_video.release()
    output_video.release()

    print("Finished writing")
    return 0


if __name__ == "__main__":
    main()
