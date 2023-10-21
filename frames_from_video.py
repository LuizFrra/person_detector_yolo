import cv2
import os

# Set the path to the video file
video_path = "./videos/patteo.mp4.mp4"

# Set the path to the output folder
output_folder = "./video_frame"

# Set the frame extraction interval (in seconds)
interval = 0.5

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame interval
frame_interval = int(interval * fps)

# Initialize the frame counter
frame_count = 0

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # If the frame counter is a multiple of the frame interval, save the frame as an image
    if frame_count % frame_interval == 0:
        # Construct the output file path
        output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")

        # Save the frame as an image
        cv2.imwrite(output_path, frame)

# Release the video file
cap.release()
