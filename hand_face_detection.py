import math

import tensorflow as tf
import numpy as np
import cv2

from utils import label_map_util
from generate_xml import AnnotationGenerator

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


def get_centroids(bouding_boxes, last_positions):
    centroids = {}
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
        centroids[class_name] = centroid

    return centroids, last_position_used


def get_normalized_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    cos_value = ((adjacent_1**2 + adjacent_2**2) -
                 opposite**2) / (2*(adjacent_1*adjacent_2))
    rad = math.acos(cos_value)

    degrees = rad / math.pi # rad * 180 to remove normalization [0 - 1]

    return degrees


def compute_triangle_features(centroids):
    triangle_features = {}

    d1, d2, d3 = compute_centroids_distances(centroids)

    triangle_features.update(
        {'distance_1': d1, 'distance_2': d2, 'distance_3': d3})

    triangle_features['perimeter'] = d1 + d2 + d3
    triangle_features['semi_perimeter'] = triangle_features['perimeter'] / 2
    triangle_features['area'] = math.sqrt(   # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
        (triangle_features['semi_perimeter'] * (triangle_features['semi_perimeter'] - d1) * (
            triangle_features['semi_perimeter'] - d2) * (triangle_features['semi_perimeter'] - d3)))

    triangle_features['height'] = 2 * triangle_features['area'] / d3

    triangle_features['ang_inter_a'] = get_normalized_angle(d3, d1, d2)
    triangle_features['ang_inter_b'] = get_normalized_angle(d1, d2, d3)
    triangle_features['ang_inter_c'] = 1 - \
        (triangle_features['ang_inter_a'] + triangle_features['ang_inter_b'])

    # teorema dos Ângulos externos https://pt.wikipedia.org/wiki/Teorema_dos_%C3%A2ngulos_externos
    triangle_features['ang_ext_a'] = triangle_features['ang_inter_b'] + \
        triangle_features['ang_inter_c']
    triangle_features['ang_ext_b'] = triangle_features['ang_inter_a'] + \
        triangle_features['ang_inter_c']
    triangle_features['ang_ext_c'] = triangle_features['ang_inter_b'] + \
        triangle_features['ang_inter_a']

    return triangle_features


def compute_features_and_draw_lines(bouding_boxes, last_positions):
    centroids, last_position_used = get_centroids(bouding_boxes, last_positions)

    triangle_features = {}
    flatten_centroids = []

    if len(centroids) == 3:
        triangle_features = compute_triangle_features(centroids)

    if len(centroids) == 3:
        class_sequence = ['hand_1', 'hand_2', 'face']
        for class_name in class_sequence:
            flatten_centroids.extend(list(centroids[class_name]))

    return triangle_features, flatten_centroids, last_position_used


def compute_centroids_distances(centroids):
    d1 = math.sqrt(
            (centroids['hand_1'][0]-centroids['face'][0])**2+(centroids['hand_1'][1]-centroids['face'][1])**2)

    d2 = math.sqrt(
            (centroids['hand_2'][0]-centroids['face'][0])**2+(centroids['hand_2'][1]-centroids['face'][1])**2)

    d3 = math.sqrt(
            (centroids['hand_1'][0]-centroids['hand_2'][0])**2+(centroids['hand_1'][1]-centroids['hand_2'][1])**2)

    return d1, d2, d3


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


def infer_images(image, label_map_path, detect_fn, heigth, width, file_name):
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

    if len(list(filter(lambda class_name: bouding_boxes[class_name] != None, bouding_boxes))) == 3:
        generate_xml = AnnotationGenerator('/home/alvaro/Documentos/video2tfrecord/object_detection_db/')
        generate_xml.generate_xml_annotation(bouding_boxes, width, heigth, file_name)
        cv2.imwrite('/home/alvaro/Documentos/video2tfrecord/object_detection_db/'+file_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image_np_with_detections, bouding_boxes


def detect_visual_cues_from_image(**kwargs):
    input_image = kwargs.get('image')

    _, bouding_boxes = infer_images(input_image, kwargs.get(
        'label_map_path'), kwargs.get('detect_fn'), kwargs.get('height'), kwargs.get('width'), kwargs.get('file_name'))

    triangle_features, centroids, last_position_used = compute_features_and_draw_lines(
        bouding_boxes, kwargs.get('last_positions'))

    face_segment, hand_1, hand_2, last_position_used = get_image_segments(
        input_image, bouding_boxes, kwargs.get('last_face_detection'),
        kwargs.get('last_hand_1_detection'), kwargs.get('last_hand_2_detection'))

    return face_segment, hand_1, hand_2, triangle_features, centroids, bouding_boxes, last_position_used
