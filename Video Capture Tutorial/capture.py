import cv2

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Get the default video frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

recording = False  # Flag to control recording

print("Press 's' to start/resume recording, 'p' to pause, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Start/Resume recording
        recording = True
        print("Recording started...")

    elif key == ord('p'):  # Pause recording
        recording = False
        print("Recording paused.")

    elif key == ord('q'):  # Quit and save
        print("Recording stopped and saved.")
        break

    if recording:
        out.write(frame)  # Write frame to file while recording

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
