import os
import math
from datetime import datetime

import tensorflow as tf
import numpy as np


def read_tfrecord(example_proto):
    feature_list = []
    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        video_stream = 'video/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        centroid_stream = 'centroid/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            video_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            centroid_stream: tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)
        feature_list.append(features)

    return feature_list


def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset


def prepare_for_training(ds, shuffle_buffer_size=100):
    ds = ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def load_data_tfrecord(tfrecord_path):
    dataset = load_dataset(tfrecord_path)

    dataset = prepare_for_training(dataset)
    return dataset


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


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


tf_record_path = tf.io.gfile.glob(
    '/home/alvaro/Desktop/video2tfrecord/example/validation/*.tfrecords')


for file_n, tfrecord in enumerate(tf_record_path):
    examples = []
    writer = None

    for example in tf.compat.v1.io.tf_record_iterator(tfrecord):
        result = tf.train.Example.FromString(example)

        for i in range(16):
            try:
                triangle_data = result.features.feature[f'triangle_data/{i}'].float_list.value
                perimeter = triangle_data[3]

                d1, d2, d3 = triangle_data[0]/perimeter, triangle_data[1] / \
                    perimeter, triangle_data[2]/perimeter

                norm_area = math.sqrt(
                    (0.5 * (0.5 - d1) * (0.5 - d2) * (0.5 - d3)))
                norm_heigth = 2 * norm_area / (d3 + 1e-10)

                new_triangle = _float_list_feature(
                    np.array([d1, d2, d3, norm_area, norm_heigth]+triangle_data[7:]))
                result.features.feature[f"triangle_data/{i}"].CopyFrom(
                    new_triangle)
            except Exception as e:
                print(e)
                new_triangle = _float_list_feature(np.array([0]*11))
                result.features.feature[f"triangle_data/{i}"].CopyFrom(
                    new_triangle)

        examples.append(result)

    writer = get_tfrecord_writer(
        '/home/alvaro/Desktop/video2tfrecord/example/val_norm', file_n, len(tf_record_path), writer)

    for rec in examples:
        writer.write(rec.SerializeToString())
