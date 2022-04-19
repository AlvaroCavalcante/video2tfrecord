import os
from itertools import combinations

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


def read_tfrecord(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    video_name = []
    triangle_stream_arr = []
    triangle_data = []
    centroids = []
    video = []

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

        triangle_data.append(tf.squeeze(tf.reshape(
            features[triangle_stream].values, (1, 13))))

        centroids.append(tf.reshape(features[centroid_stream].values, (3, 2)))

        triangle_stream_arr.append(triangle_stream)

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)
        image = get_image(features[video_stream], 512, 512)

        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)
        video.append(image)
        video_name.append(features['video_name'])
        label = tf.cast(features['label'], tf.int32)

    return [hand_1, hand_2], face, triangle_data, centroids, video, label, video_name, triangle_stream_arr


def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    # image = tf.image.per_image_standardization(image)
    image = tf.cast(image, dtype='uint8')
    return image


def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset


def prepare_for_training(ds, shuffle_buffer_size=20):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache

    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
        25).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def load_data_tfrecord(tfrecord_path):
    dataset = load_dataset(tfrecord_path)

    dataset = prepare_for_training(dataset)
    return dataset


tf_record_path = tf.io.gfile.glob(
    '/home/alvaro/Documentos/video2tfrecord/example/train/*.tfrecords')
row = 4
col = 4

dataset = load_data_tfrecord(tf_record_path)


def draw_triangle_on_img(centroids, img):
    centroids_list = [(int(centroid[0].numpy()), int(centroid[1].numpy()))
                      for centroid in centroids]

    for centroid_comb in combinations(centroids_list, 2):
        centroid_1 = centroid_comb[0]
        centroid_2 = centroid_comb[1]
        cv2.line(img, (centroid_1[0], centroid_1[1]),
                 (centroid_2[0], centroid_2[1]), (0, 255, 0), thickness=5)

    return img


def plot_figure(row, col, img_seq):
    for j in range(row*col):
        plt.subplot(row, col, j+1)
        plt.axis('off')
        plt.imshow(np.array(img_seq[j]))
    plt.show()


plot_images = True
data = []

for (hand_seq, face_seq, triangle_data, centroids, video_imgs, label, video_name_list, triangle_stream) in dataset:
    for i in range(video_name_list.shape[0]):
        if plot_images:
            plt.figure(figsize=(15, int(15*row/col)))
            plot_figure(row, col, hand_seq[i][0])
            plot_figure(row, col, hand_seq[i][1])
            plot_figure(row, col, face_seq[i])
            plot_figure(row, col, video_imgs[i])
        else:
            for j, video_name in enumerate(video_name_list[i]):
                video_name = video_name.numpy().decode('utf-8')
                video_img = video_imgs[i][j]
                img_centroids = centroids[i][j]
                triangle_img = draw_triangle_on_img(img_centroids, np.array(video_img))

                video_folder = '/home/alvaro/Documentos/video2tfrecord/results/sign1_test/'+video_name
                if not os.path.exists(video_folder):
                    os.mkdir(video_folder)

                image_name = video_folder+'/'+video_name+'_'+str(j)+'.jpg'
                
                cv2.imwrite(image_name, cv2.cvtColor(triangle_img, cv2.COLOR_BGR2RGB))
                
                data.append(list(map(lambda x: x.numpy(),
                            triangle_data[i][j][0])) + [video_name, image_name])

                df = pd.DataFrame(data, columns=['distance_1', 'distance_2', 'distance_3', 'perimeter', 'semi_perimeter', 'area', 'height',
                                'ang_inter_a', 'ang_inter_b', 'ang_inter_c', 'ang_ext_a', 'ang_ext_b', 'ang_ext_c', 'video_name', 'image_name'])
                df.to_csv(
                    '/home/alvaro/Documentos/video2tfrecord/results/sign1_test/df_res.csv', index=False)
