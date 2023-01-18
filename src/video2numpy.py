import math
import copy
import bisect

import numpy as np

import hand_face_detection
from utils import keypoint_utils
from utils import bounding_box_utils as bbox_utils
from utils import frame_processing_utils as fp_utils
from utils.stats_generator import stats
from utils import triangle_utils

HAND_WIDTH, HAND_HEIGHT = 100, 100
FACE_WIDTH, FACE_HEIGHT = 100, 100
TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT = 256, 256


def fill_data_and_convert_to_np(data, n_frames, height, width, is_image=True):
    padding_amount = n_frames - len(data)
    if padding_amount > 5:
        raise Exception('Padding amount is too high!')

    while len(data) < n_frames:
        if is_image:
            data.append(np.zeros((height, width, 3), dtype='uint8'))
        else:
            data.append([0] * width)

    return np.array(data)


def get_flatten_bbox_array(bounding_boxes):
    flatten_bbox_coords = []
    for bbox_pos in list(bounding_boxes.values()):
        flatten_bbox_coords.extend(list(bbox_pos.values()))
    return flatten_bbox_coords


def video_file_to_ndarray(i, file_path, n_frames_per_video, height, width, number_of_videos, n_channels=3):
    cap, frame_count = fp_utils.get_video_capture_and_frame_count(file_path)

    video = []
    faces = []
    hands_1 = []
    hands_2 = []
    triangle_figures = []
    facial_keypoints = []
    triangle_features_list = []
    bbox_coords = []

    last_frame = []
    last_positions = {}

    frames_counter = 0
    capture_restarted = 0
    restart = True
    rest_position = True
    rest_position_skip = 0
    frames_used = []

    position_features = {
        'hand1_position': [],
        'hand2_position': [],
        'both_hands_position': [],
        'hand1_moviment_hist': [],
        'hand2_moviment_hist': [],
        'hand1_2_moviment_hist': [],
        'both_hands_moviment_hist': []
    }

    moviment_threshold_history = []

    while restart:
        steps = int(math.floor(frame_count / n_frames_per_video))

        if frames_counter > 0:
            stop, cap, steps, capture_restarted = fp_utils.repeat_image_retrieval(
                cap, file_path, steps, capture_restarted)

            stats.repeated_videos += 1
            last_frame = []
            last_positions = {}

            if stop:
                restart = False
                break

        for frame_number in range(frame_count):
            if math.floor(frame_number % steps) == 0:
                frame = fp_utils.get_next_frame(cap)

                # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames_per_video:
                    stop, cap, steps, capture_restarted = fp_utils.repeat_image_retrieval(
                        cap, file_path, steps, capture_restarted)

                    last_frame = []
                    last_positions = {}

                    if stop:
                        restart = False

                    break

                elif frames_counter >= n_frames_per_video:
                    restart = False
                    break

                elif frame_number in frames_used:
                    continue

                else:
                    frame = fp_utils.resize_frame(
                        512, 512, 3, frame[50:650, 320:1000])

                    file_name = file_path.split(
                        '/')[-1].split('.')[0] + '_' + str(frame_number) + '.jpg'

                    face, hand_1, hand_2, triangle_features, bounding_boxes, last_positions_used = hand_face_detection.detect_visual_cues_from_image(
                        image=frame,
                        label_map_path='src/utils/assets/label_map.pbtxt',
                        height=height,
                        width=width,
                        last_frame=last_frame,
                        last_positions=last_positions,
                        file_name=file_name
                    )

                    if not triangle_features:
                        stats.missing_triangle_features += 1
                        continue

                    triangle_fig = fp_utils.resize_frame(TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT, n_channels,
                                                         triangle_utils.get_triangle_figure(
                                                             bounding_boxes, frame.shape))

                    face = fp_utils.resize_frame(
                        FACE_WIDTH, FACE_HEIGHT, n_channels, face)

                    face_keypoints = keypoint_utils.get_facial_keypoints(
                        face, last_positions, last_frame)

                    if not face_keypoints:
                        print('No face keypoints found!')
                        stats.missing_facial_keypoints += 1
                        continue

                    flatten_bbox_coords = get_flatten_bbox_array(
                        bounding_boxes)

                    temporary_position_features = copy.deepcopy(
                        position_features)

                    if not capture_restarted:
                        temporary_position_features = bbox_utils.get_position_features(
                            temporary_position_features, triangle_features)
                        temporary_position_features = bbox_utils.get_moviment_features(
                            temporary_position_features)

                        rest_position = True if len(frames_used) == 1 and temporary_position_features[
                            'both_hands_moviment_hist'][-1] < 10 else False

                        if rest_position:
                            rest_position_skip += 1
                            continue

                        if len(frames_used) == 1 and not rest_position:
                            stats.skiped_frames.append(rest_position_skip)

                        moviment_threshold_history.append(
                            temporary_position_features['both_hands_moviment_hist'][-1] < 5)

                        if len(moviment_threshold_history) >= 3 and all(moviment_threshold_history[-3:]):
                            frames_counter -= 1
                            stats.moviment_history_skip += 1
                        else:
                            video.append(fp_utils.resize_frame(
                                height, width, n_channels, frame))

                            [triangle_features.pop(key) for key in [
                                'default_distance_1', 'default_distance_2', 'default_distance_3']]
                            triangle_features_list.append(
                                list(map(lambda key: triangle_features[key], triangle_features)))

                            faces.append(face)
                            hands_1.append(fp_utils.resize_frame(
                                HAND_WIDTH, HAND_HEIGHT, n_channels, hand_1))
                            hands_2.append(fp_utils.resize_frame(
                                HAND_WIDTH, HAND_HEIGHT, n_channels, hand_2))
                            frames_used.append(frame_number)
                            bbox_coords.append(flatten_bbox_coords)
                            facial_keypoints.append(face_keypoints)
                            position_features = temporary_position_features
                            triangle_figures.append(triangle_fig)
                    else:
                        insert_index = bisect.bisect_left(
                            frames_used, frame_number)

                        temporary_position_features = bbox_utils.get_position_features(
                            temporary_position_features, triangle_features, insert_index)

                        temporary_position_features = bbox_utils.get_moviment_features(
                            temporary_position_features, insert_index)

                        moviment_threshold_history.insert(
                            insert_index, temporary_position_features['both_hands_moviment_hist'][insert_index] < 5)

                        if len(moviment_threshold_history[insert_index:insert_index+3]) >= 3 and all(moviment_threshold_history[insert_index:insert_index+3]):
                            frames_counter -= 1
                            stats.moviment_history_skip += 1
                        else:
                            video.insert(insert_index, fp_utils.resize_frame(
                                height, width, n_channels, frame))

                            [triangle_features.pop(key) for key in [
                                'default_distance_1', 'default_distance_2', 'default_distance_3']]
                            triangle_features_list.insert(insert_index, list(
                                map(lambda key: triangle_features[key], triangle_features)))
                            faces.insert(insert_index, face)
                            hands_1.insert(insert_index, fp_utils.resize_frame(
                                HAND_WIDTH, HAND_HEIGHT, n_channels, hand_1))
                            hands_2.insert(insert_index, fp_utils.resize_frame(
                                HAND_WIDTH, HAND_HEIGHT, n_channels, hand_2))
                            bbox_coords.insert(
                                insert_index, flatten_bbox_coords)
                            triangle_figures.insert(insert_index, triangle_fig)

                            facial_keypoints.insert(
                                insert_index, face_keypoints)
                            frames_used.insert(insert_index, frame_number)
                            position_features = temporary_position_features

                    last_positions = get_last_positions(
                        last_positions, position_features, bounding_boxes, last_positions_used)
                    last_frame = frame

                    frames_counter += 1
            else:
                fp_utils.get_next_frame(cap)

    print(str(i + 1) + ' of ' + str(
        number_of_videos) + ' videos within batch processed: ', file_path)

    overall_padding_amount = n_frames_per_video - len(hands_1)
    stats.padding_amount.append(overall_padding_amount)
    if overall_padding_amount > 5:
        stats.too_high_padding.append(file_path)

    faces = fill_data_and_convert_to_np(
        faces, n_frames_per_video, FACE_HEIGHT, FACE_WIDTH)
    hands_1 = fill_data_and_convert_to_np(
        hands_1, n_frames_per_video, HAND_HEIGHT, HAND_WIDTH)
    hands_2 = fill_data_and_convert_to_np(
        hands_2, n_frames_per_video, HAND_HEIGHT, HAND_WIDTH)
    video = fill_data_and_convert_to_np(
        video, n_frames_per_video, height, width)
    triangle_figures = fill_data_and_convert_to_np(
        triangle_figures, n_frames_per_video, TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT)
    triangle_features_list = fill_data_and_convert_to_np(
        triangle_features_list, n_frames_per_video, 1, 11, False)
    bbox_coords = fill_data_and_convert_to_np(
        bbox_coords, n_frames_per_video, 1, 12, False)
    hands_moviment = fill_data_and_convert_to_np(
        position_features['hand1_2_moviment_hist'], n_frames_per_video, 1, 2, False)
    facial_keypoints = fill_data_and_convert_to_np(
        facial_keypoints, n_frames_per_video, 1, 136, False)

    cap.release()
    return faces, hands_1, hands_2, triangle_features_list, bbox_coords, video, hands_moviment, facial_keypoints, triangle_figures


def get_last_positions(last_positions, position_features, bounding_boxes, last_positions_used):
    higher_moviment_hand = 'hand_1' if sum([abs(mov) for mov in position_features['hand1_moviment_hist']]) > sum(
        [abs(mov) for mov in position_features['hand2_moviment_hist']]) else 'hand_2'

    last_position_used = any(
        map(lambda pos: pos == higher_moviment_hand, last_positions_used))

    if last_position_used:
        last_positions[higher_moviment_hand] = None
    else:
        last_positions = bounding_boxes

    return last_positions


def convert_videos_to_numpy(filenames, n_frames_per_video, width, height, labels=[]):
    """Generates an ndarray from multiple video files given by filenames.
    Implementation chooses frame step size automatically for a equal separation distribution of the video images.

    Args:
      filenames: a list containing the full paths to the video files
      width: width of the video(s)
      height: height of the video(s)
      n_frames_per_video: integer value of string. Specifies the number of frames extracted from each video. If set to 'all', all frames are extracted from the
      videos and stored in the tfrecord. If the number is lower than the number of available frames, the subset of extracted frames will be selected equally
      spaced over the entire video playtime.
      n_channels: number of channels to be used for the tfrecords
      type: processing type for video data

    Returns:
      if no optical flow is used: ndarray(uint8) of shape (v,i,h,w,c) with
      v=number of videos, i=number of images, (h,w)=height and width of image,
      c=channel, if optical flow is used: ndarray(uint8) of (v,i,h,w,
      c+1)
    """

    number_of_videos = len(filenames)

    data = []
    triangle_data = []
    final_labels = []
    bbox_positions = []
    moviment_data = []
    videos = []
    error_videos = []
    facial_keypoints = []
    triangle_figures = []

    for i, file in enumerate(filenames):
        try:
            faces, hands_1, hands_2, triangle_features, centroids, video, hands_moviment, keypoints, triangle_figs = video_file_to_ndarray(i=i, file_path=file,
                                                                                                                                           n_frames_per_video=n_frames_per_video,
                                                                                                                                           height=height, width=width,
                                                                                                                                           number_of_videos=number_of_videos)
            data.append([faces, hands_1, hands_2])
            videos.append(video)
            triangle_data.append(triangle_features)
            bbox_positions.append(centroids)
            moviment_data.append(hands_moviment)
            final_labels.append(labels[i])
            facial_keypoints.append(keypoints)
            triangle_figures.append(triangle_figs)
        except Exception as e:
            print('Error to process video {}'.format(file))
            print(e)
            error_videos.append(file)
            stats.error_videos.append(file)

    return np.array(data), np.array(videos), np.array(triangle_data), np.array(bbox_positions), np.array(moviment_data), np.array(facial_keypoints), np.array(triangle_figures), final_labels, error_videos
