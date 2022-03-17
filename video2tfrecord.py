#!/usr/bin/env python
"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import pandas as pd
import math
import os
import time

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_videos_in_record', 10,
                     'Number of videos stored in one single tfrecord file')
flags.DEFINE_string('image_color_depth', "uint8",
                    'Color depth as string for the images stored in the tfrecord files. '
                    'Has to correspond to the source video color depth. '
                    'Specified as dtype (e.g. uint8 or uint16)')
flags.DEFINE_string('file_suffix', "*.mp4",
                    'defines the video file type, e.g. .mp4')

flags.DEFINE_string('source', './example/input', 'Directory with video files')
flags.DEFINE_string('destination', './example/output',
                    'Directory for storing tf records')
flags.DEFINE_integer('width_video', 1280, 'the width of the videos in pixels')
flags.DEFINE_integer('height_video', 720, 'the height of the videos in pixels')
flags.DEFINE_integer('n_frames_per_video', 5,
                     'specifies the number of frames to be taken from each video')
flags.DEFINE_integer('n_channels', 4,
                     'specifies the number of channels the videos have')
flags.DEFINE_string('video_filenames', None,
                    'specifies the video file names as a list in the case the video paths shall not be determined by the '
                    'script')


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

  return np.asarray(frame)

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
    labels.append(class_labels.loc[class_labels['video_name'] == file, ['label']].values[0][0])

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

  jpeg_encode = True
  n_channels = 3

  if video_filenames is not None:
    filenames = video_filenames
  else:
    filenames = gfile.Glob(os.path.join(source_path, file_suffix))
  if not filenames:
    raise RuntimeError('No data files found.')

  print('Total videos found: ' + str(len(filenames)))

  filenames_split = list(get_chunks(filenames, n_videos_in_record))
  class_labels = pd.read_csv(label_path, names=['video_name', 'label'])

  data = None

  for i, batch in enumerate(filenames_split):
    labels = get_data_label(batch, class_labels)

    if data is not None:
      data = None
    data, labels = convert_video_to_numpy(filenames=batch, width=width, height=height,
                                  n_frames_per_video=n_frames_per_video,
                                  n_channels=n_channels, labels=labels)
    if n_videos_in_record > len(filenames):
      total_batch_number = 1
    else:
      total_batch_number = int(math.ceil(len(filenames) / n_videos_in_record))
    print('Batch ' + str(i + 1) + '/' + str(total_batch_number) + " completed")
    assert data.size != 0, 'something went wrong during video to numpy conversion'
    save_numpy_to_tfrecords(data, batch, destination_path, 'batch_',
                            n_videos_in_record, i + 1, total_batch_number,
                            color_depth=color_depth, labels=labels, jpeg_encode=jpeg_encode)


def save_numpy_to_tfrecords(data, filenames, destination_path, name, fragmentSize,
                            current_batch_number, total_batch_number,
                            color_depth, labels, jpeg_encode=True):
  """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.

  Args:
    data: ndarray(uint8) of shape (v,i,h,w,c) with v=number of videos,
    i=number of images, c=number of image channels, h=image height, w=image
    width
    name: filename; data samples type (train|valid|test)
    fragmentSize: specifies how many videos are stored in one tfrecords file
    current_batch_number: indicates the current batch index (function call within loop)
    total_batch_number: indicates the total number of batches
    jpeg_encode: specify how to encode the video frames
  """

  num_videos = data.shape[0]
  num_images = data.shape[1]
  num_channels = data.shape[4]
  height = data.shape[2]
  width = data.shape[3]

  writer = None
  feature = {}

  for video_count in range((num_videos)):

    if video_count % fragmentSize == 0:
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

    for image_count in range(num_images):
      path = 'blob' + '/' + str(image_count)
      image = data[video_count, image_count, :, :, :]
      image = image.astype(color_depth)

      if jpeg_encode:
        image_raw = tf.image.encode_jpeg(image).numpy()
      else:
        image_raw = image.tostring()

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


def repeat_image_retrieval(cap, file_path, video, take_all_frames, steps, frame,
                           prev_frame_none, frames_counter):
  stop = False

  if frame and prev_frame_none or steps <= 0:
    stop = True
    return stop, cap, video, steps, prev_frame_none, frames_counter

  if not take_all_frames:
    # repeat with smaller step size
    steps -= 1

  prev_frame_none = True
  print("reducing step size due to error for video: ", file_path)
  frames_counter = 0
  cap.release()
  cap = get_video_capture_and_frame_count(file_path)
  # wait for image retrieval to be ready
  time.sleep(2)

  return stop, cap, video, steps, prev_frame_none, frames_counter


def video_file_to_ndarray(i, file_path, n_frames_per_video, height, width,
                          n_channels, num_real_image_channel, number_of_videos):

  cap, frame_count = get_video_capture_and_frame_count(file_path)

  take_all_frames = False
  # if not all frames are to be used, we have to skip some -> set step size accordingly
  if n_frames_per_video == 'all':
    take_all_frames = True
    video = np.zeros((frame_count, height, width, n_channels), dtype=np.uint8)
    steps = frame_count
    n_frames = frame_count
  else:
    video = np.zeros((n_frames_per_video, height, width, n_channels),
                     dtype=np.uint8)
    steps = int(math.floor(frame_count / n_frames_per_video))
    n_frames = n_frames_per_video

  assert not (frame_count < 1 or steps < 1), str(
    file_path) + " does not have enough frames. Skipping video."

  # variables needed
  image = np.zeros((height, width, num_real_image_channel),
                   dtype=FLAGS.image_color_depth)
  frames_counter = 0
  prev_frame_none = False
  restart = True
  image_prev = None
  while restart:
    for f in range(frame_count):
      if frames_counter <= 7: # skipping the first frames of the sign language video
        get_next_frame(cap)
        frames_counter += 1
        continue
      if math.floor(f % steps) == 0 or take_all_frames:
        frame = get_next_frame(cap)
        # unfortunately opencv uses bgr color format as default
        if frame is not None:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
        if frame is None and frames_counter < n_frames:
          stop, _, _1, _2, _3, _4 = repeat_image_retrieval(
              cap, file_path, video, take_all_frames, steps, frame, prev_frame_none,
              frames_counter)
          if stop:
            restart = False
            break
          else:
            video[frames_counter, :, :, :].fill(0)
            frames_counter += 1

        else:
          if frames_counter >= n_frames:
            restart = False
            break

          # iterate over channels
          for k in range(num_real_image_channel):
            resizedImage = cv2.resize(frame[:, :, k], (width, height))
            image[:, :, k] = resizedImage

          video[frames_counter, :, :, :] = image
          frames_counter += 1
      else:
        get_next_frame(cap)

  print(str(i + 1) + " of " + str(
    number_of_videos) + " videos within batch processed: ", file_path)

  video = video[8:,:,:,:]
  v = video.copy()
  cap.release()
  return v


def convert_video_to_numpy(filenames, n_frames_per_video, width, height,
                           n_channels, labels=[]):
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
  num_real_image_channel = n_channels

  data = []
  final_labels = []
  for i, file in enumerate(filenames):
    try:
      v = video_file_to_ndarray(i=i, file_path=file,
                                n_frames_per_video=n_frames_per_video,
                                height=height, width=width,
                                n_channels=n_channels,
                                num_real_image_channel=num_real_image_channel,
                                number_of_videos=number_of_videos)
      data.append(v)
      final_labels.append(labels[i])
    except Exception as e:
      print(e)

  return np.array(data), final_labels

if __name__ == '__main__':
  convert_videos_to_tfrecord(
    '/home/alvaro/Downloads/AUTSL/train', 'example/train', 
    n_videos_in_record=10, n_frames_per_video=24, file_suffix="*.mp4",
    width=800, height=600, label_path='/home/alvaro/Downloads/AUTSL/train_labels.csv')