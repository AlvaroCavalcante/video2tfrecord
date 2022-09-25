#!/usr/bin/env python
"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""
import os
import math
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from utils.stats_generator import stats

import video2numpy


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_chunks(l, n):
    """Yield successive n-sized chunks from l.
    Used to create n sublists from a list l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
            class_labels.loc[class_labels['video_name'] == int(file), ['label']].values[0][0])

    return labels


def convert_videos_to_tfrecord(source_path, destination_path,
                               n_videos_in_record=10, n_frames_per_video='all',
                               file_suffix='*.mp4',
                               width=1280, height=720,
                               video_filenames=None, label_path=None, reset_checkpoint=False):
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

    filenames = get_filenames(source_path, file_suffix, video_filenames)
    class_labels = pd.read_csv(label_path)

    filenames, checkpoint_df = recover_checkpoint(reset_checkpoint, filenames)

    total_batch_number = 1 if n_videos_in_record > len(
        filenames) else int(math.ceil(len(filenames) / n_videos_in_record))

    filenames_split = list(get_chunks(filenames, n_videos_in_record))

    for i, batch in enumerate(filenames_split):
        print('Processing batch {}'.format(str(i)))

        data = None
        labels = get_data_label(batch, class_labels)

        data, videos, triangle_data, bbox_positions, moviment_data, facial_keypoints, labels, error_videos = video2numpy.convert_videos_to_numpy(filenames=batch, width=width, height=height,
                                                                                                                                                 n_frames_per_video=n_frames_per_video, labels=labels)

        batch = list(filter(lambda file: file not in error_videos, batch))
        print('Batch ' + str(i + 1) + '/' +
              str(total_batch_number) + ' completed')
        assert data.size != 0, 'something went wrong during video to numpy conversion'

        save_numpy_to_tfrecords(data, videos, triangle_data, bbox_positions, moviment_data, facial_keypoints, batch, destination_path,
                                n_videos_in_record, i + 1, total_batch_number, labels=labels)

        checkpoint_df = save_new_checkpoint(checkpoint_df, batch, error_videos)

        stats.save_stats_as_dataframe()

def recover_checkpoint(reset_checkpoint, filenames):
    if reset_checkpoint:
        checkpoint_df = pd.DataFrame(columns=['video_name'])
    else:
        checkpoint_df = pd.read_csv('src/utils/checkpoint.csv', header=0)
        filenames = remove_from_checkpoint(checkpoint_df, filenames)
    return filenames, checkpoint_df


def save_new_checkpoint(checkpoint_df, batch, error_videos):
    print('Saving new video checkpoint')

    for bt_file in batch:
        checkpoint_df = checkpoint_df.append(
            {'video_name': bt_file}, ignore_index=True)
    checkpoint_df.to_csv('src/utils/checkpoint.csv', index=False)
    return checkpoint_df


def get_filenames(source_path, file_suffix, video_filenames):
    if video_filenames is not None:
        filenames = video_filenames
    else:
        filenames = gfile.Glob(os.path.join(source_path, file_suffix))
    if not filenames:
        raise RuntimeError('No data files found.')

    print('Total videos found: ' + str(len(filenames)))
    return filenames


def remove_from_checkpoint(checkpoint_df, filenames):
    for file in list(checkpoint_df['video_name'].values):
        if file in filenames:
            filenames.remove(file)
    return filenames


def save_numpy_to_tfrecords(data, videos, triangle_data, bbox_positions, moviment_data, facial_keypoints, filenames, destination_path, fragment_size,
                            current_batch_number, total_batch_number, labels):
    """Converts an entire dataset into x tfrecords where x=videos/fragment_size.

    Args:
      data: ndarray(uint8) of shape (v,i,h,w,c) with v=number of videos,
      i=number of images, c=number of image channels, h=image height, w=image
      width
      name: filename; data samples type (train|valid|test)
      fragment_size: specifies how many videos are stored in one tfrecords file
      current_batch_number: indicates the current batch index (function call within loop)
      total_batch_number: indicates the total number of batches
    """

    print('Starting TF Record Write')
    num_videos = data.shape[0]
    num_images = data.shape[2]
    num_channels = data.shape[5]
    height = data.shape[3]
    width = data.shape[4]

    writer = None
    feature = {}

    for video_count in range((num_videos)):

        if video_count % fragment_size == 0:
            writer = get_tfrecord_writer(
                destination_path, current_batch_number, total_batch_number, writer)

        for image_count in range(num_images):
            face_stream = 'face/' + str(image_count)
            hand_1_stream = 'hand_1/' + str(image_count)
            hand_2_stream = 'hand_2/' + str(image_count)
            triangle_stream = 'triangle_data/' + str(image_count)
            video_stream = 'video/' + str(image_count)
            bbox_stream = 'bbox/' + str(image_count)
            moviment_stream = 'moviment/' + str(image_count)
            keypoint_stream = 'keypoint/' + str(image_count)

            face_image = data[video_count, 0,
                              image_count, :, :, :].astype('uint8')
            hand_1_image = data[video_count, 1,
                                image_count, :, :, :].astype('uint8')
            hand_2_image = data[video_count, 2,
                                image_count, :, :, :].astype('uint8')
            video_img = videos[video_count,
                               image_count, :, :, :].astype('uint8')

            face_raw = tf.image.encode_jpeg(
                face_image).numpy()
            hand_1_raw = tf.image.encode_jpeg(
                hand_1_image).numpy()
            hand_2_raw = tf.image.encode_jpeg(
                hand_2_image).numpy()
            video_img_raw = tf.image.encode_jpeg(
                video_img).numpy()

            file = filenames[video_count].split('/')[-1].split('.')[0]
            file = '_'.join(file.split('_')[0:2])

            feature[face_stream] = _bytes_feature(face_raw)
            feature[hand_1_stream] = _bytes_feature(hand_1_raw)
            feature[hand_2_stream] = _bytes_feature(hand_2_raw)
            feature[video_stream] = _bytes_feature(video_img_raw)

            feature['video_name'] = _bytes_feature(str.encode(file))
            feature['height'] = _int64_feature(height)
            feature['width'] = _int64_feature(width)
            feature['depth'] = _int64_feature(num_channels)
            feature['label'] = _int64_feature(labels[video_count])

            feature[bbox_stream] = _float_list_feature(
                bbox_positions[video_count][image_count])

            feature[triangle_stream] = _float_list_feature(
                triangle_data[video_count][image_count])

            feature[moviment_stream] = _float_list_feature(
                moviment_data[video_count][image_count])

            feature[keypoint_stream] = _float_list_feature(
                facial_keypoints[video_count][image_count])

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    if writer is not None:
        writer.close()


def get_tfrecord_writer(destination_path, current_batch_number, total_batch_number, writer):
    if writer is not None:
        writer.close()
    filename = os.path.join(destination_path,
                            'batch_' + str(current_batch_number) + '_of_' + str(
                                total_batch_number) + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%s') + '.tfrecords')
    print('Writing', filename)
    if tf.__version__.split('.')[0] == '2':
        writer = tf.io.TFRecordWriter(filename)
    else:
        writer = tf.python_io.TFRecordWriter(filename)
    return writer


if __name__ == '__main__':
    convert_videos_to_tfrecord(
        '/home/alvaro/Downloads/DATASETS/wlasl_val', 'results/wlasl_val',
        n_videos_in_record=360, n_frames_per_video=16, file_suffix='*.mp4',
        width=256, height=256, label_path='/home/alvaro/Downloads/DATASETS/wlasl_val/labels.csv', reset_checkpoint=True)
