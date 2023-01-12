import os
import random
from itertools import combinations

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

from data_augmentation import transform_image


def get_apply_proba_dict():
    apply_proba_dict = {}
    aug_keys = ['brightness', 'contrast', 'saturation', 'hue',
                'flip_left_right', 'rotation', 'shear', 'zoom', 'shift']

    apply_proba_dict = {}

    for key in aug_keys:
        apply_proba_dict[key] = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    return apply_proba_dict


def get_range_aug_dict(img_width):
    range_aug_dict = {}

    rotation_range = [-20, 20]
    shear_range = [5, 12]
    h_zoom_range = [0.8, 1.2]
    w_zoom_range = [0.8, 1.2]
    h_shift_range = [0, 0.15]
    w_shift_range = [0, 0.05]

    range_aug_dict['rotation'] = tf.random.uniform([1], rotation_range[0],
                                                   rotation_range[1], dtype=tf.float32)
    range_aug_dict['shear'] = tf.random.uniform([1], shear_range[0],
                                                shear_range[1], dtype=tf.float32)
    range_aug_dict['height_zoom'] = tf.random.uniform(
        [1], h_zoom_range[0], h_zoom_range[1], dtype=tf.float32)
    range_aug_dict['width_zoom'] = tf.random.uniform(
        [1], w_zoom_range[0], w_zoom_range[1], dtype=tf.float32)
    range_aug_dict['height_shift'] = tf.random.uniform(
        [1], h_shift_range[0], h_shift_range[1], dtype=tf.float32) * img_width
    range_aug_dict['width_shift'] = tf.random.uniform(
        [1], w_shift_range[0], w_shift_range[1], dtype=tf.float32) * img_width

    return range_aug_dict


def read_tfrecord(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    hands = []
    video_name = []
    triangle_stream_arr = []
    triangle_data = []
    bouding_boxes = []
    video = []
    keypoints = []
    triangle_figures = []

    apply_proba_dict = get_apply_proba_dict()
    range_aug_dict = get_range_aug_dict(80)
    seed = random.randint(0, 10000)

    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        video_stream = 'video/' + str(image_count)
        triangle_fig_stream = 'tri_figures/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        bbox_stream = 'bbox/' + str(image_count)
        moviment_stream = 'moviment/' + str(image_count)
        keypoint_stream = 'keypoint/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            video_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_fig_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            bbox_stream: tf.io.VarLenFeature(tf.float32),
            moviment_stream: tf.io.VarLenFeature(tf.float32),
            keypoint_stream: tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)

        moviment = tf.squeeze(
            tf.reshape(features[moviment_stream].values, (1, 2)))

        triangle = tf.squeeze(tf.reshape(
            features[triangle_stream].values, (1, 11)))

        triangle_data.append(tf.concat([triangle, moviment], axis=0))

        bouding_boxes.append(tf.reshape(features[bbox_stream].values, (1, 12)))

        keypoints.append(tf.squeeze(tf.reshape(
            features[keypoint_stream].values, (1, 136))))

        triangle_stream_arr.append(triangle_stream)

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)
        image = get_image(features[video_stream], 512, 512)
        triangle_fig = get_image(features[triangle_fig_stream], 256, 256)

        # face_image = transform_image(
        #     face_image, width, apply_proba_dict, range_aug_dict, seed)
        hand_1_image = transform_image(
            hand_1_image, width, apply_proba_dict, range_aug_dict, seed, True)
        hand_2_image = transform_image(
            hand_2_image, width, apply_proba_dict, range_aug_dict, seed, True)

        face.append(face_image)
        # hand_1.append(hand_1_image)
        # hand_2.append(hand_2_image)
        hands.append(tf.concat([hand_1_image, hand_2_image], axis=1))
        triangle_figures.append(triangle_fig)
        video.append(image)
        video_name.append(features['video_name'])
        label = tf.cast(features['label'], tf.int32)

    return hands, face, triangle_data, bouding_boxes, video, label, video_name, triangle_stream_arr, keypoints,triangle_figures


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


def prepare_for_training(ds, shuffle_buffer_size=30):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache

    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
        25).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def filter_func(hands, face, triangle_data, centroids, video, label, video_name, triangle_stream_arr):
    return tf.math.greater(label, 206)


def load_data_tfrecord(tfrecord_path):
    dataset = load_dataset(tfrecord_path)
    # dataset = dataset.filter(filter_func)

    dataset = prepare_for_training(dataset)
    return dataset


tf_record_path = tf.io.gfile.glob(
    '/home/alvaro/Desktop/video2tfrecord/results/train_v6/*.tfrecords')
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


def plot_figure(row, col, img_seq, centroids=None, triangle=False, keypoints=[]):
    for j in range(row*col):
        plt.subplot(row, col, j+1)
        plt.axis('off')
        if triangle:
            triangle_img = draw_triangle_on_img(
                centroids[j], np.array(img_seq[j]))
            plt.imshow(triangle_img)
        else:
            img = np.array(img_seq[j])
            if len(keypoints):
                for i in range(0, len(keypoints[j]), 2):
                    cv2.circle(
                        img, (keypoints[j][i].numpy(), keypoints[j][i+1].numpy()), 2, (0, 255, 0), -1)

            plt.imshow(img)
    plt.show()


plot_images = True
data = []

# pred_df = pd.read_csv(
#     '/home/alvaro/Desktop/multi-cue-sign-language/top3_predictions_lstm.csv')

# pred_incorrect = pred_df[pred_df['correct_prediction'] == False]
# pred_incorrect = list(pred_incorrect['video_names'].values)

for (hand_seq, face_seq, triangle_data, bboxes, video_imgs, label, video_name_list, triangle_stream, keypoints, triangle_figures) in dataset:
    for i in range(video_name_list.shape[0]):
        if plot_images:
            # plot_figure(row, col, video_imgs[i], bboxes[i], True)
            plot_figure(row, col, triangle_figures[i])
            plot_figure(row, col, hand_seq[i])
            # plot_figure(row, col, face_seq[i], keypoints=keypoints[i])
        else:
            for j, video_name in enumerate(video_name_list[i]):
                video_name = video_name.numpy().decode('utf-8')
                video_img = video_imgs[i][j]
                img_centroids = bboxes[i][j]
                triangle_img = draw_triangle_on_img(
                    img_centroids, np.array(video_img))

                video_folder = '/home/alvaro/Documentos/video2tfrecord/results/sign1_test/'+video_name
                if not os.path.exists(video_folder):
                    os.mkdir(video_folder)

                image_name = video_folder+'/'+video_name+'_'+str(j)+'.jpg'

                cv2.imwrite(image_name, cv2.cvtColor(
                    triangle_img, cv2.COLOR_BGR2RGB))

                data.append(list(map(lambda x: x.numpy(),
                            triangle_data[i][j][0])) + [video_name, image_name])

                df = pd.DataFrame(data, columns=['distance_1', 'distance_2', 'distance_3', 'perimeter', 'semi_perimeter', 'area', 'height',
                                                 'ang_inter_a', 'ang_inter_b', 'ang_inter_c', 'ang_ext_a', 'ang_ext_b', 'ang_ext_c', 'video_name', 'image_name'])
                df.to_csv(
                    '/home/alvaro/Documentos/video2tfrecord/results/sign1_test/df_res.csv', index=False)
