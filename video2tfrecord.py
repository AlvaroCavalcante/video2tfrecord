#!/usr/bin/env python
"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""
import os
import time
import math
import bisect

import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.platform import gfile

import hand_face_detection

MODEL = tf.saved_model.load(
        '/home/alvaro/Downloads/centernet_resnet50_v2-20220318T025013Z-001/centernet_resnet50_v2/saved_model')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_chunks(l, n):
    """Yield successive n-sized chunks from l.
    Used to create n sublists from a list l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


def get_data_label(batch_files, class_labels):
    """Gets the video labels based on the name of the video. Considering
    that we have a dataframe (from csv) that contains the video_name and label
    columns, we can get the label based on a split of the video name.

    Args:
      batch_files: list of file paths in the batch.
      class_labels: dataframe containing the name of the videos and the respective label.
    """
    labels = []
    for file in batch_files:
        file = file.split('/')[-1].split('.')[0]
        file = '_'.join(file.split('_')[0:2])
        labels.append(
            class_labels.loc[class_labels['video_name'] == file, ['label']].values[0][0])

    return labels


def convert_videos_to_tfrecord(source_path, destination_path,
                               n_videos_in_record=10, n_frames_per_video='all',
                               file_suffix="*.mp4",
                               width=1280, height=720,
                               color_depth="uint8", video_filenames=None, label_path=None):
    """Starts the process of converting video files to tfrecord files.

    Args:
      source_path: directory where video videos are stored

      destination_path: directory where tfrecords should be stored

      n_videos_in_record: Number of videos stored in one single tfrecord file

      n_frames_per_video: integer value of string. Specifies the number of frames extracted from each video. If set to 'all', all frames are extracted from the
        videos and stored in the tfrecord. If the number is lower than the number of available frames, the subset of extracted frames will be selected equally
        spaced over the entire video playtime.

      file_suffix: defines the video file type, e.g. *.mp4

      width: the width of the videos in pixels

      height: the height of the videos in pixels

      color_depth: Color depth as string for the images stored in the tfrecord
        files. Has to correspond to the source video color depth. Specified as
        dtype (e.g. uint8 or uint16)

      video_filenames: specify, if the the full paths to the videos can be
        directly be provided. In this case, the source will be ignored.
    """
    assert isinstance(n_frames_per_video, (int, str))
    if type(n_frames_per_video) is str:
        assert n_frames_per_video == "all"

    if video_filenames is not None:
        filenames = video_filenames
    else:
        filenames = gfile.Glob(os.path.join(source_path, file_suffix))
    if not filenames:
        raise RuntimeError('No data files found.')

    print('Total videos found: ' + str(len(filenames)))

    filenames_split = list(get_chunks(filenames, n_videos_in_record))
    class_labels = pd.read_csv(label_path, names=['video_name', 'label'])

    total_batch_number = 1 if n_videos_in_record > len(
        filenames) else int(math.ceil(len(filenames) / n_videos_in_record))

    data = None

    for i, batch in enumerate(filenames_split):
        labels = get_data_label(batch, class_labels)

        if data is not None:
            data = None
        data, labels = convert_video_to_numpy(filenames=batch, width=width, height=height,
                                              n_frames_per_video=n_frames_per_video, labels=labels)
        
        print('Batch ' + str(i + 1) + '/' +
              str(total_batch_number) + " completed")
        assert data.size != 0, 'something went wrong during video to numpy conversion'

        save_numpy_to_tfrecords(data, batch, destination_path, 'batch_',
                                n_videos_in_record, i + 1, total_batch_number,
                                color_depth=color_depth, labels=labels)


def save_numpy_to_tfrecords(data, filenames, destination_path, name, fragmentSize,
                            current_batch_number, total_batch_number,
                            color_depth, labels):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.

    Args:
      data: ndarray(uint8) of shape (v,i,h,w,c) with v=number of videos,
      i=number of images, c=number of image channels, h=image height, w=image
      width
      name: filename; data samples type (train|valid|test)
      fragmentSize: specifies how many videos are stored in one tfrecords file
      current_batch_number: indicates the current batch index (function call within loop)
      total_batch_number: indicates the total number of batches
    """

    num_videos = data.shape[0]
    num_images = data.shape[2]
    num_channels = data.shape[5]
    height = data.shape[3]
    width = data.shape[4]

    writer = None
    feature = {}

    for video_count in range((num_videos)):

        if video_count % fragmentSize == 0:
            writer = get_tfrecord_writer(destination_path, name, current_batch_number, total_batch_number, writer)

        for image_count in range(num_images):
            path = 'blob' + '/' + str(image_count)
            image = data[video_count, image_count, :, :, :]
            image = image.astype(color_depth)

            image_raw = tf.image.encode_jpeg(image).numpy()  # image.tostring()

            file = filenames[video_count].split('/')[-1].split('.')[0]
            file = '_'.join(file.split('_')[0:2])

            feature[path] = _bytes_feature(image_raw)
            feature['video_name'] = _bytes_feature(str.encode(file))
            feature['height'] = _int64_feature(height)
            feature['width'] = _int64_feature(width)
            feature['depth'] = _int64_feature(num_channels)
            feature['label'] = _int64_feature(labels[video_count])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    if writer is not None:
        writer.close()

def get_tfrecord_writer(destination_path, name, current_batch_number, total_batch_number, writer):
    if writer is not None:
        writer.close()
    filename = os.path.join(destination_path,
                                    name + str(current_batch_number) + '_of_' + str(
                                        total_batch_number) + '.tfrecords')
    print('Writing', filename)
    if tf.__version__.split('.')[0] == '2':
        writer = tf.io.TFRecordWriter(filename)
    else:
        writer = tf.python_io.TFRecordWriter(filename)
    return writer


def repeat_image_retrieval(cap, file_path, take_all_frames, steps, capture_restarted):
    stop = False

    if capture_restarted or steps <= 0:
        stop = True
        return stop, cap, steps, capture_restarted

    if not take_all_frames:
        # repeat with smaller step size
        steps -= 1

    capture_restarted = True
    print("reducing step size due to error for video: ", file_path)
    cap.release()
    cap, _ = get_video_capture_and_frame_count(file_path)
    # wait for image retrieval to be ready
    time.sleep(2)

    return stop, cap, steps, capture_restarted


def video_file_to_ndarray(i, file_path, n_frames_per_video, height, width, number_of_videos, n_channels=3):
    cap, frame_count = get_video_capture_and_frame_count(file_path)

    take_all_frames = True if n_frames_per_video == 'all' else False

    n_frames = frame_count if take_all_frames else n_frames_per_video

    video = []
    faces = []
    hands_1 = []
    hands_2 = []

    last_face_detection = []
    last_hand_1_detection = []
    last_hand_2_detection = []
    last_positions = {}
    last_position_used = False

    # variables needed
    frames_counter = 0
    capture_restarted = False
    restart = True
    frames_used = []

    while restart:
        frames_to_skip = 8
        
        steps = frame_count if take_all_frames else int(
            math.floor((frame_count - frames_to_skip) / n_frames_per_video))

        assert not (frame_count < 1 or steps < 1), str(
            file_path) + " does not have enough frames. Skipping video."

        if frames_counter > 0:
            stop, cap, steps, capture_restarted = repeat_image_retrieval(
                cap, file_path, take_all_frames, steps, capture_restarted)

            if stop:
                restart = False 
                break

        for frame_number in range(frame_count):
            if frames_to_skip > 0:  # skipping the first frames of the sign language video
                get_next_frame(cap)
                frames_to_skip -= 1
                continue
            if math.floor(frame_number % steps) == 0 or take_all_frames:
                frame = get_next_frame(cap)

                # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames:
                    stop, cap, steps, capture_restarted = repeat_image_retrieval(
                        cap, file_path, take_all_frames, steps, capture_restarted)

                    if stop:
                        restart = False

                    break

                elif frames_counter >= n_frames:
                    restart = False
                    break

                elif frame_number in frames_used:
                    continue

                else:
                    face, hand_1, hand_2, triangle_features, bouding_boxes, last_position_used = hand_face_detection.detect_visual_cues_from_image(
                        image=frame,
                        label_map_path='utils/label_map.pbtxt',
                        detect_fn=MODEL,
                        height=512,
                        width=512,
                        last_face_detection=last_face_detection,
                        last_hand_1_detection=last_hand_1_detection,
                        last_hand_2_detection=last_hand_2_detection,
                        last_positions=last_positions
                    )

                    if not triangle_features:
                        continue

                    # iterate over channels
                    # TODO: why to resize each channel individually?
                    
                    if not capture_restarted:
                        video.append(resize_frame(height, width, n_channels, frame))
                        faces.append(resize_frame(50, 50, n_channels, face))
                        hands_1.append(resize_frame(50, 50, n_channels, hand_1))
                        hands_2.append(resize_frame(50, 50, n_channels, hand_2))
                    else:
                        insert_index = bisect.bisect_left(frames_used, frame_number)
                        video.insert(insert_index, resize_frame(height, width, n_channels, frame))
                        faces.insert(insert_index, resize_frame(50, 50, n_channels, face))
                        hands_1.insert(insert_index, resize_frame(50, 50, n_channels, hand_1))
                        hands_2.insert(insert_index, resize_frame(50, 50, n_channels, hand_2))

                    last_face_detection = [] if last_position_used else faces[frames_counter]
                    last_hand_1_detection = [] if last_position_used else hands_1[frames_counter]
                    last_hand_2_detection = [] if last_position_used else hands_2[frames_counter]
                    last_positions = {} if last_position_used else bouding_boxes
                    last_position_used = False

                    frames_counter += 1
                    frames_used.append(frame_number)
            else:
                get_next_frame(cap)

    print(str(i + 1) + " of " + str(
        number_of_videos) + " videos within batch processed: ", file_path)

    video = np.array(video)
    faces = np.array(faces)
    hands_1 = np.array(hands_1)
    hands_2 = np.array(hands_2)

    cap.release()
    return faces, hands_1, hands_2

def resize_frame(height, width, n_channels, frame):
    image = np.zeros((height, width, n_channels), dtype='uint8')

    for n_channel in range(n_channels):
        resized_image = cv2.resize(
                            frame[:, :, n_channel], (width, height))
        image[:, :, n_channel] = resized_image

    return image


def convert_video_to_numpy(filenames, n_frames_per_video, width, height, labels=[]):
    """Generates an ndarray from multiple video files given by filenames.
    Implementation chooses frame step size automatically for a equal separation distribution of the video images.

    Args:
      filenames: a list containing the full paths to the video files
      width: width of the video(s)
      height: height of the video(s)
      n_frames_per_video: integer value of string. Specifies the number of frames extracted from each video. If set to 'all', all frames are extracted from the
      videos and stored in the tfrecord. If the number is lower than the number of available frames, the subset of extracted frames will be selected equally
      spaced over the entire video playtime.
      n_channels: number of channels to be used for the tfrecords
      type: processing type for video data

    Returns:
      if no optical flow is used: ndarray(uint8) of shape (v,i,h,w,c) with
      v=number of videos, i=number of images, (h,w)=height and width of image,
      c=channel, if optical flow is used: ndarray(uint8) of (v,i,h,w,
      c+1)
    """

    number_of_videos = len(filenames)

    data = []
    final_labels = []
    for i, file in enumerate(filenames):
        try:
            faces, hands_1, hands_2 = video_file_to_ndarray(i=i, file_path=file,
                                                            n_frames_per_video=n_frames_per_video,
                                                            height=height, width=width,
                                                            number_of_videos=number_of_videos)
            data.append([faces, hands_1, hands_2])
            final_labels.append(labels[i])
        except Exception as e:
            print(e)

    return np.array(data), final_labels


if __name__ == '__main__':
    convert_videos_to_tfrecord(
        '/home/alvaro/Downloads/AUTSL/train', 'example/train',
        n_videos_in_record=1, n_frames_per_video=24, file_suffix="*.mp4",
        width=800, height=600, label_path='/home/alvaro/Downloads/AUTSL/train_labels.csv')
