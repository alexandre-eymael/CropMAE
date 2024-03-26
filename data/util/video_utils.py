import random
from decord import VideoReader
import decord
import numpy as np
import torch

# Set decord to use torch
decord.bridge.set_bridge('torch')

def sample_frames(video_path, gap_range=[4, 48], repeated_sampling=1):
    """
    Sample a pair of frames from a video with a random gap between them.
    """

    vr = VideoReader(video_path)
    n_frames = len(vr)

    if n_frames < 2:
        raise ValueError("Video must have at least 2 frames")

    # Set gap and sample 2 indices
    lower_bound = min(gap_range[0], n_frames - 1)
    upper_bound = min(gap_range[1], n_frames)
    gaps = np.random.randint(lower_bound, upper_bound, repeated_sampling)

    indices = []
    for gap in gaps:
        frame1_idx = random.randint(0, n_frames - gap - 1)
        frame2_idx = frame1_idx + gap
        indices.extend([frame1_idx, frame2_idx])

    # Extract frames and permute to (N, C, H, W) torch format
    frames = vr.get_batch(indices).permute(0, 3, 1, 2)

    # Reform pairs of frames
    frames = torch.chunk(frames, repeated_sampling, dim=0)

    return frames

def sample_frame(video_path, repeated_sampling=1):
    """
    Sample a frame from a video.
    """
    
    vr = VideoReader(video_path)
    n_frames = len(vr)

    if n_frames < 1:
        raise ValueError("Video must have at least 1 frame")

    # Sample indices
    indices = np.random.randint(0, n_frames, repeated_sampling)

    # Extract frames and permute to (N, C, H, W) torch format
    frames = vr.get_batch(indices).permute(0, 3, 1, 2)

    return frames

if __name__ == "__main__":

    vid = "test.mp4"
    for frame1, frame2 in sample_frames(vid, gap_range=[4, 48], repeated_sampling=1):
        print(frame1.shape, frame2.shape)