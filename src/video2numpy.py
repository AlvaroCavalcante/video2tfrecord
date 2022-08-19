import math
import copy
import bisect

import numpy as np

import hand_face_detection
from utils import bounding_box_utils as bbox_utils
from utils import frame_processing_utils as fp_utils


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
    hand_width, hand_height = 100, 100
    face_width, face_height = 100, 100

    cap, frame_count = fp_utils.get_video_capture_and_frame_count(file_path)

    take_all_frames = True if n_frames_per_video == 'all' else False

    n_frames = frame_count if take_all_frames else n_frames_per_video

    video = []
    faces = []
    hands_1 = []
    hands_2 = []
    triangle_features_list = []
    bbox_coords = []

    last_frame = []
    last_positions = {}
    last_position_used = False

    frames_counter = 0
    capture_restarted = False
    restart = True
    frames_used = []

    position_features = {
        'hand1_position': [],
        'hand2_position': [],
        'both_hands_position': [],
        'hand1_moviment_hist': [],
        'hand2_moviment_hist': [],
        'both_hands_moviment_hist': []
    }

    moviment_threshold_history = []

    while restart:
        # initial frames to skip in sign language video
        frames_to_skip = fp_utils.get_frames_skip(frame_count)

        steps = frame_count if take_all_frames else int(
            math.floor((frame_count - frames_to_skip) / n_frames_per_video))

        if frames_counter > 0:
            stop, cap, steps, capture_restarted = fp_utils.repeat_image_retrieval(
                cap, file_path, take_all_frames, steps, capture_restarted)

            if stop:
                restart = False
                break

        for frame_number in range(frame_count):
            if frames_to_skip > 0:  # skipping the first frames of the sign language video
                fp_utils.get_next_frame(cap)
                frames_to_skip -= 1
                continue
            if math.floor(frame_number % steps) == 0 or take_all_frames:
                frame = fp_utils.get_next_frame(cap)

                # special case handling: opencv's frame count sometimes differs from real frame count -> repeat
                if frame is None and frames_counter < n_frames:
                    stop, cap, steps, capture_restarted = fp_utils.repeat_image_retrieval(
                        cap, file_path, take_all_frames, steps, capture_restarted)

                    if stop:
                        restart = False

                    break

                elif frames_counter >= n_frames:
                    restart = False
                    break

                elif frame_number in frames_used:
                    continue

                else:
                    file_name = file_path.split(
                        '/')[-1].split('.')[0] + '_' + str(frame_number) + '.jpg'

                    face, hand_1, hand_2, triangle_features, bounding_boxes, last_position_used = hand_face_detection.detect_visual_cues_from_image(
                        image=frame,
                        label_map_path='src/utils/label_map.pbtxt',
                        height=height,
                        width=width,
                        last_frame=last_frame,
                        last_positions=last_positions,
                        file_name=file_name
                    )

                    if not triangle_features:
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

                        moviment_threshold_history.append(
                            temporary_position_features['both_hands_moviment_hist'][-1] < 5)

                        if len(moviment_threshold_history) >= 3 and all(moviment_threshold_history[-3:]):
                            frames_counter -= 1
                        else:
                            video.append(fp_utils.resize_frame(
                                height, width, n_channels, frame))
                            triangle_features_list.append(
                                list(map(lambda key: triangle_features[key], triangle_features)))
                            faces.append(fp_utils.resize_frame(
                                face_width, face_height, n_channels, face))
                            hands_1.append(fp_utils.resize_frame(
                                hand_width, hand_height, n_channels, hand_1))
                            hands_2.append(fp_utils.resize_frame(
                                hand_width, hand_height, n_channels, hand_2))
                            frames_used.append(frame_number)
                            bbox_coords.append(flatten_bbox_coords)
                            position_features = temporary_position_features
                    else:
                        insert_index = bisect.bisect_left(
                            frames_used, frame_number)

                        temporary_position_features = bbox_utils.get_position_features(
                            temporary_position_features, triangle_features, insert_index)

                        temporary_position_features = bbox_utils.get_moviment_features(
                            temporary_position_features, insert_index)

                        moviment_threshold_history.append(
                            temporary_position_features['both_hands_moviment_hist'][insert_index] < 5)

                        if len(moviment_threshold_history[0:insert_index]) > 3 and all(moviment_threshold_history[insert_index-3:insert_index]):
                            frames_counter -= 1
                        else:
                            video.insert(insert_index, fp_utils.resize_frame(
                                height, width, n_channels, frame))
                            triangle_features_list.insert(insert_index, list(
                                map(lambda key: triangle_features[key], triangle_features)))
                            faces.insert(insert_index, fp_utils.resize_frame(
                                face_width, face_height, n_channels, face))
                            hands_1.insert(insert_index, fp_utils.resize_frame(
                                hand_width, hand_height, n_channels, hand_1))
                            hands_2.insert(insert_index, fp_utils.resize_frame(
                                hand_width, hand_height, n_channels, hand_2))
                            bbox_coords.insert(
                                insert_index, flatten_bbox_coords)

                            frames_used.insert(insert_index, frame_number)
                            position_features = temporary_position_features

                    last_frame = [] if last_position_used else frame
                    last_positions = {} if last_position_used else bounding_boxes
                    last_position_used = False

                    frames_counter += 1
            else:
                fp_utils.get_next_frame(cap)

    print(str(i + 1) + ' of ' + str(
        number_of_videos) + ' videos within batch processed: ', file_path)

    faces = fill_data_and_convert_to_np(
        faces, n_frames, face_height, face_width)
    hands_1 = fill_data_and_convert_to_np(
        hands_1, n_frames, hand_height, hand_width)
    hands_2 = fill_data_and_convert_to_np(
        hands_2, n_frames, hand_height, hand_width)
    video = fill_data_and_convert_to_np(video, n_frames, height, width)
    triangle_features_list = fill_data_and_convert_to_np(
        triangle_features_list, n_frames, 1, 11, False)
    bbox_coords = fill_data_and_convert_to_np(
        bbox_coords, n_frames, 1, 12, False)

    cap.release()
    return faces, hands_1, hands_2, triangle_features_list, bbox_coords, video


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
    centroids_positions = []
    videos = []
    error_videos = []

    for i, file in enumerate(filenames):
        try:
            faces, hands_1, hands_2, triangle_features, centroids, video = video_file_to_ndarray(i=i, file_path=file,
                                                                                                 n_frames_per_video=n_frames_per_video,
                                                                                                 height=height, width=width,
                                                                                                 number_of_videos=number_of_videos)
            data.append([faces, hands_1, hands_2])
            videos.append(video)
            triangle_data.append(triangle_features)
            centroids_positions.append(centroids)
            final_labels.append(labels[i])
        except Exception as e:
            print('Error to process video {}'.format(file))
            print(e)
            error_videos.append(file)

    return np.array(data), np.array(videos), np.array(triangle_data), np.array(centroids_positions), final_labels, error_videos
