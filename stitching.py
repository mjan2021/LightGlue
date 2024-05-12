# import cv2
# import torch
# from lightglue import SuperPoint, LightGlue
# from lightglue.utils import rbd

# # Initialize extractor and matcher
# extractor = SuperPoint(max_num_keypoints=2048).eval()
# matcher = LightGlue(features='superpoint').eval()

# # Initialize video capture objects
# cap0 = cv2.VideoCapture('assets/video0.mp4')
# cap1 = cv2.VideoCapture('assets/video1.mp4')

# # Get video properties
# fps = cap0.get(cv2.CAP_PROP_FPS)
# width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Initialize video writer
# out = cv2.VideoWriter('assets/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

# while True:
#     ret0, frameOrg0 = cap0.read()
#     ret1, frameOrg1 = cap1.read()

#     if not (ret0 and ret1):
#         break  # Break the loop if one of the videos ends

#     # Convert frames to PyTorch tensors
#     image0 = torch.Tensor(frameOrg0).permute(2, 0, 1).unsqueeze(0)
#     image1 = torch.Tensor(frameOrg1).permute(2, 0, 1).unsqueeze(0)

#     # Extract features and match
#     feats0 = extractor.extract(image0)
#     feats1 = extractor.extract(image1)
#     matches01 = matcher({"image0": feats0, "image1": feats1})
#     feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

#     # Get matched keypoints
#     kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
#     m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

#     # Draw matches on the frames
#     # matched_frame = cv2.drawMatches(frameOrg0, kpts0, frameOrg1, kpts1, matches1to2=matches, outImg=None)

#     # Combine frames horizontally
#     combined_frame = cv2.hconcat([frameOrg0, frameOrg1])

#     # Write the combined frame to the output video
#     out.write(combined_frame)

#     # Display the combined frame
#     cv2.imshow('Stitched Video', combined_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and writer objects
# cap0.release()
# cap1.release()
# out.release()
# cv2.destroyAllWindows()
import datetime
import cv2
import torch
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
start = datetime.datetime.now()
# Load the videos
cap0 = cv2.VideoCapture('assets/video0.mp4')
cap1 = cv2.VideoCapture('assets/video1.mp4')

# Initialize SuperPoint feature extractor and LightGlue matcher
extractor = DISK(max_num_keypoints=2048).eval()
matcher = LightGlue(features='disk').eval()

# Create video writer for the output
output_width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = max(int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap0.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('assets/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Process frames from both videos
counter = 0
while cap0.isOpened():
    counter += 1
    ret0, frame_org0 = cap0.read()
    ret1, frame_org1 = cap1.read()

    if not ret0 or not ret1:
        break

    # Convert frames to PyTorch tensors
    frame0 = torch.tensor(cv2.cvtColor(frame_org0, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0).float()
    frame1 = torch.tensor(cv2.cvtColor(frame_org1, cv2.COLOR_BGR2GRAY)).unsqueeze(0).unsqueeze(0).float()

    # Extract features
    feats0 = extractor.extract(frame0)
    feats1 = extractor.extract(frame1)

    # Match features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Get matched keypoints
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    print(f'total matches: {len(matches)}')
    
    # Find homography
    H, _ = cv2.findHomography(m_kpts1.cpu().numpy(), m_kpts0.cpu().numpy(), cv2.RANSAC)

    # Warp the second frame
    stitched_frame = cv2.warpPerspective(frame_org1, H, (output_width, output_height))
    
    # Create the final stitched frame
    stitched_frame[:, 0:frame_org0.shape[1]] = frame_org0
    
    # Combine frames horizontally and draw matching features
    # combined_frame = cv2.hconcat([frame_org0, frame_org1])
    # for i in range(len(m_kpts0)):
    #     pt1 = (int(m_kpts0[i][0]), int(m_kpts0[i][1]))
    #     pt2 = (int(m_kpts1[i][0] + frame_org0.shape[1]), int(m_kpts1[i][1]))
    #     cv2.line(combined_frame, pt1, pt2, (0, 255, 0), 1)

    # Write to output
    out.write(stitched_frame)
    cv2.imwrite('assets/stitched_frame.jpg', stitched_frame)
    # Display the combined frame with matched features
    cv2.imshow('Combined Video with Matched Features', stitched_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap0.release()
cap1.release()
out.release()
cv2.destroyAllWindows()
end = datetime.datetime.now()
t = end-start
print(f'total time elapsed: {t.seconds()}s')
