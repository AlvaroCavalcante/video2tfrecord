import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

def read_tfrecord(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    triangle_img = []
    video_name = []
    triangle_stream_arr = []
    triangle_data = []
    centroids = []

    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        triangle_images_stream = 'triangle_images_data/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_images_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            'centroids': tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)

        triangle_data.append(tf.reshape(features[triangle_stream].values, (1, 13)))
        centroids.append(tf.reshape(features['centroids'].values, (3, 2)))

        triangle_stream_arr.append(triangle_stream)
        
        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)
        triangle_image = get_image(features[triangle_images_stream], 512, 512)

        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)
        triangle_img.append(triangle_image)

        video_name.append(features['video_name'])
        label = tf.cast(features['label'], tf.int32)

    return [hand_1, hand_2], face, triangle_data, triangle_img, centroids, label, video_name, triangle_stream_arr


def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    image = tf.cast(image, dtype='uint8')
    return image


def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset


def prepare_for_training(ds, shuffle_buffer_size=5):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache
    # ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
        # 4).prefetch(tf.data.experimental.AUTOTUNE)

    ds = ds.repeat().batch(17).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def load_data_tfrecord(tfrecord_path):
    dataset = load_dataset(tfrecord_path)

    dataset = prepare_for_training(dataset)
    return dataset


tf_record_path = tf.io.gfile.glob(
    '/home/alvaro/Documentos/video2tfrecord/example/train/batch_1_of_9.tfrecords')
row = 4
col = 4
#row = min(row,15//col)

all_elements = load_data_tfrecord(tf_record_path).unbatch()
augmented_element = all_elements.repeat().batch(17)


def plot_figure(row, col, img_seq):
    for j in range(row*col):
        plt.subplot(row, col, j+1)
        plt.axis('off')
        plt.imshow(np.array(img_seq[j]))
    plt.show()

data = []

for (hand_seq, face_seq, triangle_data, triangle_image, centroids, label, video_name, triangle_stream) in augmented_element:
    for i in range(video_name.shape[0]):
        for j, video in enumerate(video_name[i]):
            video = video.numpy().decode('utf-8')
            video_folder = '/home/alvaro/Documentos/video2tfrecord/results/sign_1/'+video
            if not os.path.exists(video_folder):
                os.mkdir(video_folder)

            image_name = video_folder+'/'+video+'_'+str(j)+'.jpg'
            cv2.imwrite(image_name, cv2.cvtColor(np.array(triangle_image[i][j]), cv2.COLOR_BGR2RGB))
            data.append(list(map(lambda x: x.numpy(), triangle_data[i][j][0])) + [video, image_name])
    break
    # plt.figure(figsize=(15, int(15*row/col)))
    # plot_figure(row, col, hand_seq[0][0])
    # plot_figure(row, col, hand_seq[0][1])
    # plot_figure(row, col, face_seq[0])
    # plot_figure(row, col, triangle_image[0])

df = pd.DataFrame(data, columns=['distance_1', 'distance_2', 'distance_3', 'perimeter', 'semi_perimeter', 'area', 'height', 'ang_inter_a', 'ang_inter_b', 'ang_inter_c', 'ang_ext_a', 'ang_ext_b', 'ang_ext_c', 'video_name', 'image_name'])
df.to_csv('/home/alvaro/Documentos/video2tfrecord/results/sign_1/df_res.csv', index=False)
