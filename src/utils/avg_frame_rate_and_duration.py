"""
This script iterates over the dataset to calculate
the mean frame rate and duration of the videos.
"""

import os
import cv2

def with_opencv(filename):
    video = cv2.VideoCapture(filename)

    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = total_num_frames / frame_rate

    return duration, total_num_frames

vid_path = '/home/alvaro/Documents/AUTSL_VIDEO_DATA/train/train'

videos = os.listdir(vid_path)
frames = []
durations = []


for i, video in enumerate(videos):
    duration, frame = with_opencv(f'{vid_path}/{video}')
    
    frames.append(frame)
    durations.append(duration)

    print(f'Remaining videos: {len(videos)-i}')

print('Mean frames:', sum(frames)/len(frames))
print('Mean duration:', sum(durations)/len(durations))
