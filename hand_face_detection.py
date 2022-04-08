import math
from itertools import combinations

import tensorflow as tf
import numpy as np
import cv2

from utils import label_map_util

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


def get_centroids(bouding_boxes, last_positions):
    centroids = []
    last_position_used = False

    for class_name in bouding_boxes:
        bbox = bouding_boxes.get(class_name)
        if bbox:
            xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
        else:
            bbox = last_positions.get(class_name)
            last_position_used = True

            if not bbox:
                return centroids, last_position_used

            xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']

        centroid = (int((xmin+xmax)/2), int((ymin+ymax)/2))
        centroids.append(centroid)

    return centroids, last_position_used


def get_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    cos_value = ((adjacent_1**2 + adjacent_2**2) -
                 opposite**2) / (2*(adjacent_1*adjacent_2))
    rad = math.acos(cos_value)

    degrees = rad * 180 / math.pi

    return degrees


def compute_triangle_features(centroids, img, draw_on_img):
    euclidian_distances, img = compute_centroids_distances(
        img, draw_on_img, centroids)

    d1, d2, d3 = euclidian_distances[0], euclidian_distances[1], euclidian_distances[2]
    triangle_features = {}

    triangle_features.update(
        {'distance_1': d1, 'distance_2': d2, 'distance_3': d3})

    triangle_features['perimeter'] = d1 + d2 + d3
    triangle_features['semi_perimeter'] = triangle_features['perimeter'] / 2
    triangle_features['area'] = math.sqrt(   # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
        (triangle_features['semi_perimeter'] * (triangle_features['semi_perimeter'] - d1) * (
            triangle_features['semi_perimeter'] - d2) * (triangle_features['semi_perimeter'] - d3)))

    triangle_features['ang_inter_a'] = get_angle(d3, d1, d2)
    triangle_features['ang_inter_b'] = get_angle(d1, d2, d3)
    triangle_features['ang_inter_c'] = 180.0 - \
        (triangle_features['ang_inter_a'] + triangle_features['ang_inter_b'])

    # teorema dos Ângulos externos https://pt.wikipedia.org/wiki/Teorema_dos_%C3%A2ngulos_externos
    triangle_features['ang_ext_a'] = triangle_features['ang_inter_b'] + \
        triangle_features['ang_inter_c']
    triangle_features['ang_ext_b'] = triangle_features['ang_inter_a'] + \
        triangle_features['ang_inter_c']
    triangle_features['ang_ext_c'] = triangle_features['ang_inter_b'] + \
        triangle_features['ang_inter_a']

    return triangle_features, img


def compute_features_and_draw_lines(bouding_boxes, img, last_positions, draw_on_img=True):
    centroids, last_position_used = get_centroids(bouding_boxes, last_positions)

    triangle_features = {}
    if len(centroids) == 3:
        triangle_features, img = compute_triangle_features(
            centroids, img, draw_on_img)

    return img, triangle_features, last_position_used


def compute_centroids_distances(img, draw_on_img, centroids):
    euclidian_distances = []
    for centroid_comb in combinations(centroids, 2):
        centroid_1 = centroid_comb[0]
        centroid_2 = centroid_comb[1]
        if draw_on_img:
            cv2.line(img, (centroid_1[0], centroid_1[1]),
                     (centroid_2[0], centroid_2[1]), (0, 255, 0), thickness=5)
        distance = math.sqrt(
            (centroid_1[0]-centroid_2[0])**2+(centroid_1[1]-centroid_2[1])**2)
        euclidian_distances.append(distance)

    return euclidian_distances, img


def filter_boxes_and_draw(image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width, draw_on_image=False):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                                        use_display_name=True)

    output_bboxes = {'face': None, 'hand_1': None, 'hand_2': None}
    hand_counter = 2
    face_counter = 1
    for i in np.where(scores > .4)[0]:
        class_name = category_index[classes[i]].get('name')

        if face_counter == 0 and hand_counter == 0:
            return image_np_with_detections, output_bboxes
        elif class_name == 'face' and face_counter == 0:
            continue
        elif class_name == 'hand' and hand_counter == 0:
            continue

        class_name = 'hand_' + \
            str(hand_counter) if class_name == 'hand' else class_name

        xmin, xmax, ymin, ymax = boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2]
        xmin, xmax, ymin, ymax = int(
            xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)
        output_bboxes[class_name] = {'xmin': xmin,
                                     'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

        if draw_on_image:
            color = (0, 255, 0) if class_name == 'face' else (255, 0, 0)
            cv2.rectangle(image_np_with_detections,
                          (xmin, ymin), (xmax, ymax), color, 2)

        if class_name == 'face':
            face_counter -= 1
        else:
            hand_counter -= 1

    return image_np_with_detections, output_bboxes


def get_image_segments(input_image, bouding_boxes, last_face_detection, last_hand_1_detection, last_hand_2_detection):
    face = None
    hand_1 = None
    hand_2 = None
    img_segment = None
    last_position_used = False

    for class_name in bouding_boxes:
        bbox = bouding_boxes.get(class_name)

        if bbox:
            img_segment = input_image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
        else:
            last_position_used = True

        if class_name == 'face':
            face = img_segment if not last_position_used else last_face_detection
        elif class_name == 'hand_1':
            hand_1 = img_segment if not last_position_used else last_hand_1_detection
        else:
            hand_2 = img_segment if not last_position_used else last_hand_2_detection

    return face, hand_1, hand_2, last_position_used


def infer_images(image, label_map_path, detect_fn, heigth, width):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    image_np_with_detections, bouding_boxes = filter_boxes_and_draw(
        image.copy(),
        label_map_path,
        detections['detection_scores'],
        detections['detection_classes'],
        detections['detection_boxes'],
        heigth, width)

    return image_np_with_detections, bouding_boxes


def detect_visual_cues_from_image(**kwargs):
    input_image = kwargs.get('image')

    drawn_image, bouding_boxes = infer_images(input_image, kwargs.get(
        'label_map_path'), kwargs.get('detect_fn'), kwargs.get('height'), kwargs.get('width'))

    drawn_image, triangle_features, last_position_used = compute_features_and_draw_lines(
        bouding_boxes, drawn_image, kwargs.get('last_positions'))

    face_segment, hand_1, hand_2, last_position_used = get_image_segments(
        input_image, bouding_boxes, kwargs.get('last_face_detection'), 
        kwargs.get('last_hand_1_detection'), kwargs.get('last_hand_2_detection'))

    return face_segment, hand_1, hand_2, triangle_features, drawn_image, bouding_boxes, last_position_used
