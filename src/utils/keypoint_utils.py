import cv2
import dlib
from imutils import face_utils

KEYPOINT_PREDICTOR = dlib.shape_predictor('src/utils/shape_predictor_68_face_landmarks.dat')


def keypoint_inference(face_img, face_rect):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    dlib_prediction = KEYPOINT_PREDICTOR(gray_face, face_rect)
    facial_keypoints = face_utils.shape_to_np(dlib_prediction)
    return facial_keypoints


def get_last_face_frame(last_positions, last_frame):
    if not len(last_positions):
        return last_frame

    bbox = last_positions.get('face')
    last_face = last_frame[bbox['ymin']
                :bbox['ymax'], bbox['xmin']:bbox['xmax']]

    return last_face


def get_facial_keypoints(face, last_positions, last_frame):
    face_rect = dlib.rectangle(left=0, top=0, right=100, bottom=100)
    keypoints_list = []

    facial_keypoints = keypoint_inference(face, face_rect)

    if not len(facial_keypoints) and len(last_positions):
        last_face = get_last_face_frame(last_positions, last_frame)
        facial_keypoints = keypoint_inference(last_face, face_rect)

    if len(facial_keypoints):
        for (x, y) in facial_keypoints:
            keypoints_list.extend([x, y])

    return keypoints_list
