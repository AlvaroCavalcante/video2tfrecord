import math

import cv2
import numpy as np


def get_centroids(bouding_boxes):
    centroids = {}

    for class_name in bouding_boxes:
        bbox = bouding_boxes.get(class_name)
        if bbox:
            xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
        else:
            return centroids

        centroid = (int((xmin+xmax)/2), int((ymin+ymax)/2))
        centroids[class_name] = centroid

    return centroids


def get_normalized_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    try:
        cos_value = ((adjacent_1**2 + adjacent_2**2) -
                     opposite**2) / (2*(adjacent_1*adjacent_2) + 1e-10)
        rad = math.acos(cos_value)

        degrees = rad / math.pi  # rad * 180 to remove normalization [0 - 1]

        return degrees
    except Exception as e:
        print('Error to calculate normalized angle')
        print(e)


def compute_centroids_distances(centroids):
    try:
        d1 = math.sqrt(
            (centroids['hand_1'][0]-centroids['face'][0])**2+(centroids['hand_1'][1]-centroids['face'][1])**2)

        d2 = math.sqrt(
            (centroids['hand_2'][0]-centroids['face'][0])**2+(centroids['hand_2'][1]-centroids['face'][1])**2)

        d3 = math.sqrt(
            (centroids['hand_1'][0]-centroids['hand_2'][0])**2+(centroids['hand_1'][1]-centroids['hand_2'][1])**2)

        return d1, d2, d3
    except Exception as e:
        print('Error to calculate centroids distances')
        print(e)


def compute_triangle_features(centroids):
    try:
        triangle_features = {}

        d1, d2, d3 = compute_centroids_distances(centroids)
        triangle_features.update(
            {'default_distance_1': d1, 'default_distance_2': d2, 'default_distance_3': d3})

        perimeter = d1 + d2 + d3
        norm_semi_perimeter = 0.5  # considering that perimeter is 1

        d1, d2, d3 = d1/perimeter, d2/perimeter, d3/perimeter

        triangle_features.update(
            {'distance_1': d1, 'distance_2': d2, 'distance_3': d3})

        triangle_features['area'] = math.sqrt(   # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
            (norm_semi_perimeter * (norm_semi_perimeter - d1) * (
                norm_semi_perimeter - d2) * (norm_semi_perimeter - d3)))

        # avoid 0 division
        triangle_features['height'] = 2 * \
            triangle_features['area'] / (d3 + 1e-10)

        triangle_features['ang_inter_a'] = get_normalized_angle(d3, d1, d2)
        triangle_features['ang_inter_b'] = get_normalized_angle(d1, d2, d3)
        triangle_features['ang_inter_c'] = 1 - \
            (triangle_features['ang_inter_a'] +
             triangle_features['ang_inter_b'])

        # teorema dos Ângulos externos https://pt.wikipedia.org/wiki/Teorema_dos_%C3%A2ngulos_externos
        triangle_features['ang_ext_a'] = triangle_features['ang_inter_b'] + \
            triangle_features['ang_inter_c']
        triangle_features['ang_ext_b'] = triangle_features['ang_inter_a'] + \
            triangle_features['ang_inter_c']
        triangle_features['ang_ext_c'] = triangle_features['ang_inter_b'] + \
            triangle_features['ang_inter_a']

        return triangle_features
    except Exception as e:
        print('Error to calculate triangle features')
        print(e)


def draw_triangle_on_img(centroids, triangle_figure):
    points = np.array([list(centroids[cent]) for cent in centroids])
    cv2.fillPoly(triangle_figure, pts=[points], color=(255, 255, 255))

    return triangle_figure


def get_triangle_figure(bounding_boxes, input_img_shape):
    centroids = get_centroids(bounding_boxes)
    triangle_figure = np.zeros(input_img_shape, np.uint8)

    triangle_figure = draw_triangle_on_img(centroids, triangle_figure)

    for class_name in centroids:
        if class_name == 'face':
            cv2.circle(triangle_figure,
                       centroids[class_name], 20, (0, 255, 0), -1)

        xmin, xmax, ymin, ymax = get_reduced_bbox(
            bounding_boxes[class_name])

        if class_name == 'hand_1':
            cv2.rectangle(triangle_figure, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), -1)
        elif class_name == 'hand_2':
            points = np.array(
                [[(xmin + xmax)//2, ymin], [xmax, ymax], [xmin, ymax]])
            cv2.fillPoly(triangle_figure, pts=[points], color=(255, 0, 0))

    return triangle_figure


def get_reduced_bbox(boxes):
    xmin, xmax, ymin, ymax = boxes['xmin'], boxes['xmax'], boxes['ymin'], boxes['ymax']

    xmin += int(0.15 * (xmax - xmin))
    xmax -= int(0.15 * (xmax - xmin))
    ymin += int(0.15 * (ymax - ymin))
    ymax -= int(0.15 * (ymax - ymin))
    return xmin, xmax, ymin, ymax
