import os
import time

import cv2
import numpy as np


def resize_frame(height, width, n_channels, frame):
    image = np.zeros((height, width, n_channels), dtype='uint8')

    for n_channel in range(n_channels):
        resized_image = cv2.resize(
            frame[:, :, n_channel], (width, height))
        image[:, :, n_channel] = resized_image

    return image


def repeat_image_retrieval(cap, file_path, take_all_frames, steps, capture_restarted):
    stop = False

    if not take_all_frames:
        # repeat with smaller step size
        steps -= 1

    if capture_restarted or steps <= 0:
        stop = True
        return stop, cap, steps, capture_restarted

    capture_restarted = True
    print('reducing step size due to error for video: ', file_path)
    cap.release()
    cap, _ = get_video_capture_and_frame_count(file_path)
    # wait for image retrieval to be ready
    time.sleep(.5)

    return stop, cap, steps, capture_restarted


def get_video_capture_and_frame_count(path):
    assert os.path.isfile(
        path), "Couldn't find video file:" + path + ". Skipping video."
    cap = None
    if path:
        cap = cv2.VideoCapture(path)

    assert cap is not None, "Couldn't load video capture:" + path + ". Skipping video."

    # compute meta data of video
    if hasattr(cv2, 'cv'):
        frame_count = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, frame_count


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None

    frame = np.asarray(frame)
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
