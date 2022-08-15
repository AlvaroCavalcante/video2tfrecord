import math 

import tensorflow as tf
import numpy as np
import cv2
from scipy.spatial import distance as dist

from utils import label_map_util
from utils import triangle_utils
from generate_xml import AnnotationGenerator

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


def compute_features_and_draw_lines(bouding_boxes):
    centroids = triangle_utils.get_centroids(bouding_boxes)

    triangle_features = {}

    if len(centroids) == 3:
        triangle_features = triangle_utils.compute_triangle_features(centroids)

    return triangle_features


def filter_boxes_and_draw(image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width, last_positions, draw_on_image=False):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                                        use_display_name=True)

    output_bboxes = {'face': None, 'hand_1': None, 'hand_2': None}
    hand_counter = 2
    face_counter = 1

    for index in np.where(scores > .55)[0]:
        class_name = category_index[classes[index]].get('name')

        if face_counter == 0 and hand_counter == 0:
            return image_np_with_detections, output_bboxes
        elif class_name == 'face' and face_counter == 0:
            continue
        elif class_name == 'hand' and hand_counter == 0:
            continue

        class_name = 'hand_' + \
            str(hand_counter) if class_name == 'hand' else class_name

        xmin, xmax, ymin, ymax = get_box_coordinates(
            boxes, heigth, width, index)

        class_name = define_class_name({'xmin': xmin,
                                        'xmax': xmax, 'ymin': ymin, 'ymax': ymax}, last_positions, class_name)

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


def define_class_name(current_positions, last_positions, class_name):
    if not last_positions or class_name == 'face':
        return class_name

    current_centroid = (int((current_positions['xmin']+current_positions['xmax'])/2), int(
        (current_positions['ymin']+current_positions['ymax'])/2))

    last_centroids = [(int((last_positions[hand]['xmin']+last_positions[hand]['xmax'])/2), int(
        (last_positions[hand]['ymin']+last_positions[hand]['ymax'])/2)) for hand in ['hand_1', 'hand_2']]

    distance = dist.cdist(last_centroids, [current_centroid])


    d1 = math.sqrt(
            (current_centroid[0]-last_centroids[0][0])**2+(current_centroid[1]-last_centroids[0][1])**2)

    d2 = math.sqrt(
            (current_centroid[0]-last_centroids[1][0])**2+(current_centroid[1]-last_centroids['hand_2'][1])**2)

    print(distance)

    return class_name    


def get_box_coordinates(boxes, heigth, width, index):
    xmin, xmax, ymin, ymax = boxes[index][1], boxes[index][3], boxes[index][0], boxes[index][2]
    xmin, xmax, ymin, ymax = int(
        xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)

    # the code bellow could be used to increase the bouding box size by a percentage.
    xmin -= int(0.10 * (xmax - xmin))
    xmax += int(0.10 * (xmax - xmin))
    ymin -= int(0.10 * (ymax - ymin))
    ymax += int(0.10 * (ymax - ymin))

    return xmin, xmax, ymin, ymax


def get_image_segments(input_image, bouding_boxes, last_frame, last_positions):
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
            bbox = last_positions.get(class_name)
            if bbox:
                img_segment = last_frame[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
                bouding_boxes[class_name] = bbox

        if class_name == 'face':
            face = img_segment
        elif class_name == 'hand_1':
            hand_1 = img_segment
        else:
            hand_2 = img_segment

    return face, hand_1, hand_2, last_position_used, bouding_boxes


def infer_images(image, label_map_path, detect_fn, heigth, width, file_name, last_positions):
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
        generate_xml = AnnotationGenerator('./object_detection_db_autsl/')
        generate_xml.generate_xml_annotation(
            bouding_boxes, width, heigth, file_name)
        cv2.imwrite('./object_detection_db_autsl/'+file_name,
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite('./errors_db_autsl/'+file_name,
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image_np_with_detections, bouding_boxes


def detect_visual_cues_from_image(**kwargs):
    input_image = kwargs.get('image')

    _, bouding_boxes = infer_images(input_image, kwargs.get(
        'label_map_path'), kwargs.get('detect_fn'), kwargs.get('height'), kwargs.get('width'), kwargs.get('file_name'), kwargs.get('last_positions'))

    face_segment, hand_1, hand_2, last_position_used, bouding_boxes = get_image_segments(
        input_image, bouding_boxes, kwargs.get('last_frame'), kwargs.get('last_positions'))

    triangle_features = compute_features_and_draw_lines(bouding_boxes)

    return face_segment, hand_1, hand_2, triangle_features, bouding_boxes, last_position_used
