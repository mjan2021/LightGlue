import cv2
import numpy as np
import math

# Function to calculate endpoint based on angle and length
def calculate_endpoint(start_point, angle_degrees, curve_length):
    angle_radians = math.radians(angle_degrees)
    end_point_x = int(start_point[0] + curve_length * math.cos(angle_radians))
    end_point_y = int(start_point[1] - curve_length * math.sin(angle_radians))  # Minus sin for y (due to inverted y-axis in images)
    return (end_point_x, end_point_y)

# Load video file
video_path = "./video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create video writer to save output
output_path = "output_video.avi"
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Parameters for drawing the curve
start_point = (100, 100)
angle_degrees = 45.0  # Initial angle in degrees
curve_length = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate endpoint based on current angle and length
    end_point = calculate_endpoint(start_point, angle_degrees, curve_length)

    # Draw the curve line on the frame
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, cv2.LINE_AA)

    # Display the frame with the curve line
    cv2.imshow("Video with Curve Line", frame)
    out.write(frame)  # Write frame to output video

    # Update angle for next frame (example: increasing by 1 degree per frame)
    angle_degrees += 1

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

