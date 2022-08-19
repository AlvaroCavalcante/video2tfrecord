import math

import cv2
import numpy as np

from generate_xml import AnnotationGenerator
from utils import label_map_util


def auto_annotate_images(image, heigth, width, file_name, bouding_boxes):
    generate_xml = AnnotationGenerator('./object_detection_db_autsl/')
    generate_xml.generate_xml_annotation(
        bouding_boxes, width, heigth, file_name)
    cv2.imwrite('./object_detection_db_autsl/'+file_name,
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def align_class_names(current_positions, last_positions):
    if not last_positions:
        return current_positions

    current_centroids = [(int((current_positions[hand]['xmin']+current_positions[hand]['xmax'])/2), int(
        (current_positions[hand]['ymin']+current_positions[hand]['ymax'])/2)) for hand in ['hand_1', 'hand_2']]

    last_centroids = [(int((last_positions[hand]['xmin']+last_positions[hand]['xmax'])/2), int(
        (last_positions[hand]['ymin']+last_positions[hand]['ymax'])/2)) for hand in ['hand_1', 'hand_2']]

    # current1_hand2 = math.sqrt(
    #     (current_centroids[0][0]-last_centroids[1][0])**2+(current_centroids[0][1]-last_centroids[1][1])**2)

    # current2_hand2 = math.sqrt(
    #     (current_centroids[1][0]-last_centroids[1][0])**2+(current_centroids[1][1]-last_centroids[1][1])**2)

    current_hand1_last_hand1 = math.sqrt(
        (current_centroids[0][0]-last_centroids[0][0])**2+(current_centroids[0][1]-last_centroids[0][1])**2)

    current_hand2_last_hand1 = math.sqrt(
        (current_centroids[1][0]-last_centroids[0][0])**2+(current_centroids[1][1]-last_centroids[0][1])**2)

    if current_hand1_last_hand1 < current_hand2_last_hand1:
        return current_positions
    else:
        return {'hand_1': current_positions['hand_2'], 'hand_2': current_positions['hand_1'], 'face': current_positions['face']}


def filter_boxes_and_draw(image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width, draw_on_image=False):
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


def get_position_features(position_features, triangle_features, insert_index=None):
    """
    This function calculates the position of the hands in relation
    to the face (fixed point in image). Besides that, it also calculates
    the absolute distance between both hands in relation to the face, in order
    to check if there's any moviment in the frame.
    """

    both_hands_position = triangle_features['default_distance_1'] + \
        triangle_features['default_distance_2']

    hand_1_position = triangle_features['default_distance_1']
    hand_2_position = triangle_features['default_distance_2']

    if not insert_index:
        position_features['both_hands_position'].append(both_hands_position)
        position_features['hand1_position'].append(hand_1_position)
        position_features['hand2_position'].append(hand_2_position)
    else:
        position_features['both_hands_position'].insert(
            insert_index, both_hands_position)
        position_features['hand1_position'].insert(
            insert_index, hand_1_position)
        position_features['hand2_position'].insert(
            insert_index, hand_2_position)

    return position_features


def get_moviment_features(position_features, insert_index=None):
    if len(position_features['both_hands_position']) > 1:
        if not insert_index:
            both_hands_mov = abs(
            position_features['both_hands_position'][-1] -
            position_features['both_hands_position'][-2]
            )

            hand1_mov = (position_features['hand1_position'][-1] - \
                position_features['hand1_position'][-2]) / max(position_features['hand1_position'])

            hand2_mov = (position_features['hand2_position'][-1] - \
                position_features['hand2_position'][-2]) / max(position_features['hand2_position'])

            position_features['both_hands_moviment_hist'].append(
                both_hands_mov)
            position_features['hand1_moviment_hist'].append(hand1_mov)
            position_features['hand2_moviment_hist'].append(hand2_mov)
        else:
            both_hands_mov = abs(
            position_features['both_hands_position'][insert_index] -
            position_features['both_hands_position'][insert_index-1]
            )

            hand1_mov = (position_features['hand1_position'][insert_index] - \
                position_features['hand1_position'][insert_index-1]) / max(position_features['hand1_position'])

            hand2_mov = (position_features['hand2_position'][insert_index] - \
                position_features['hand2_position'][insert_index-1]) / max(position_features['hand1_position'])

            position_features['both_hands_moviment_hist'].insert(
                insert_index, both_hands_mov)
            position_features['hand1_moviment_hist'].insert(
                insert_index, hand1_mov)
            position_features['hand2_moviment_hist'].insert(
                insert_index, hand2_mov)
    else:
        position_features['both_hands_moviment_hist'].append(0)
        position_features['hand1_moviment_hist'].append(0)
        position_features['hand2_moviment_hist'].append(0)

    return position_features
