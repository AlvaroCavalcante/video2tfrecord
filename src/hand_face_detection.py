import tensorflow as tf
import numpy as np
import cv2

from utils import triangle_utils
from utils import bounding_box_utils as bbox_utils
from utils.stats_generator import stats


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


MODEL = tf.saved_model.load(
    '/home/alvaro/Desktop/hand-face-detector/signer_independent_model/saved_model')


def compute_features_and_draw_lines(bouding_boxes):
    centroids = triangle_utils.get_centroids(bouding_boxes)

    triangle_features = {}

    if len(centroids) == 3:
        triangle_features = triangle_utils.compute_triangle_features(centroids)

    return triangle_features


def get_image_segments(input_image, bouding_boxes, last_frame, last_positions_use):
    face = None
    hand_1 = None
    hand_2 = None
    img_segment = None

    for index, class_name in enumerate(bouding_boxes):
        bbox = bouding_boxes.get(class_name)

        if bbox and not last_positions_use[index]:
            img_segment = input_image[bbox['ymin']
                :bbox['ymax'], bbox['xmin']:bbox['xmax']]
        elif bbox and last_positions_use[index]:
            img_segment = last_frame[bbox['ymin']
                :bbox['ymax'], bbox['xmin']:bbox['xmax']]

        if class_name == 'face':
            face = img_segment
        elif class_name == 'hand_1':
            hand_1 = img_segment
        else:
            hand_2 = img_segment

    return face, hand_1, hand_2


def infer_images(image, label_map_path, heigth, width, file_name):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = MODEL(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    image_np_with_detections, bouding_boxes = bbox_utils.filter_boxes_and_draw(
        image.copy(),
        label_map_path,
        detections['detection_scores'],
        detections['detection_classes'],
        detections['detection_boxes'],
        heigth, width)

    if len(list(filter(lambda class_name: bouding_boxes[class_name] != None, bouding_boxes))) == 3:
        stats.correct_detections += 1
        bbox_utils.auto_annotate_images(image, heigth, width, file_name, bouding_boxes)
    else:
        stats.missing_detections += 1
        cv2.imwrite('./errors_db_autsl/'+file_name,
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image_np_with_detections, bouding_boxes


def check_last_position_use(bounding_boxes, last_positions):
    """
    This method checks if there's any missing bounding box in the dict. If this
    is the case, then we try to use the last bounding box of the previous frame.
    """
    last_positions_used = []

    for class_name in bounding_boxes:
        bbox = bounding_boxes.get(class_name)
        if bbox:
            last_positions_used.append('')
        else:
            bbox = last_positions.get(class_name)
            if bbox:
                bounding_boxes[class_name] = bbox
                last_positions_used.append(class_name)
            else:
                last_positions_used.append('')

    return bounding_boxes, last_positions_used


def detect_visual_cues_from_image(**kwargs):
    input_image = kwargs.get('image')

    _, bounding_boxes = infer_images(input_image, kwargs.get(
        'label_map_path'), kwargs.get('height'), kwargs.get('width'), kwargs.get('file_name'))

    bounding_boxes, last_positions_used = check_last_position_use(bounding_boxes, kwargs.get('last_positions'))
    bounding_boxes = bbox_utils.align_class_names(bounding_boxes, kwargs.get('last_positions'))

    face_segment, hand_1, hand_2 = get_image_segments(
        input_image, bounding_boxes, kwargs.get('last_frame'), last_positions_used)

    triangle_features = compute_features_and_draw_lines(bounding_boxes)

    return face_segment, hand_1, hand_2, triangle_features, bounding_boxes, last_positions_used
