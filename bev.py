import cv2
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# Load the videos
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4']
caps = [cv2.VideoCapture(path) for path in video_paths]

# Initialize SuperPoint feature extractor and LightGlue matcher
extractor = SuperPoint(max_num_keypoints=2048).eval()
matcher = LightGlue(features='superpoint').eval()

# Get video properties
fps = caps[0].get(cv2.CAP_PROP_FPS)
width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for the output
output_width = width * 2  # Adjust as needed
output_height = height * 2  # Adjust as needed
out = cv2.VideoWriter('birdseye_view.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Initialize variables to hold homographies
num_videos = len(video_paths)
homographies = [np.eye(3)] * num_videos

# Process frames from all videos
while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    if len(frames) < num_videos:
        break
    
    # Convert frames to PyTorch tensors
    frame_tensors = [torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0).float() for frame in frames]
    
    # Extract features and match keypoints
    feats = [extractor.extract(frame) for frame in frame_tensors]
    matches = [matcher({'image0': feats[i], 'image1': feats[(i+1)%num_videos]}) for i in range(num_videos)]
    feats = [rbd(feats[i]) for i in range(num_videos)]
    matches = [rbd(matches[i]) for i in range(num_videos)]
    
    # Estimate homographies between adjacent videos
    for i in range(num_videos):
        kpts0, kpts1, matches01 = feats[i]['keypoints'], feats[(i+1)%num_videos]['keypoints'], matches[i]['matches']
        m_kpts0, m_kpts1 = kpts0[matches01[..., 0]], kpts1[matches01[..., 1]]
        H, _ = cv2.findHomography(m_kpts1.cpu().numpy(), m_kpts0.cpu().numpy(), cv2.RANSAC)
        homographies[(i+1)%num_videos] = np.dot(H, homographies[i])
    
    # Warp frames and blend to create bird's eye view
    birdseye_view = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        warped_frame = cv2.warpPerspective(frame, homographies[i], (output_width, output_height))
        birdseye_view += warped_frame // num_videos
    
    # Write to output
    out.write(birdseye_view)
    
    # Display the bird's eye view
    cv2.imshow('Birds Eye View', birdseye_view)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()
