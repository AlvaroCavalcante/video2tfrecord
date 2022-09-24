import pickle
import time

class DatasetStatistics():

    def __init__(self) -> None:
        self.skiped_frames = []
        self.missing_facial_keypoints = 0
        self.missing_triangle_features = 0
        self.moviment_history_skip = 0
        self.too_high_padding = []
        self.padding_amount = []
        self.correct_detections = 0
        self.missing_detections = 0
        self.error_videos = []
        self.repeated_videos = 0

    def save_stats_as_dataframe(self):
        val_object = {
            'skiped_frames': self.skiped_frames,
            'missing_facial_keypoints': self.missing_facial_keypoints,
            'missing_triangle_features': self.missing_triangle_features,
            'moviment_history_skip': self.moviment_history_skip,
            'too_high_padding': self.too_high_padding,
            'padding_amount': self.padding_amount,
            'correct_detections': self.correct_detections,
            'missing_detections': self.missing_detections,
            'error_videos': self.error_videos,
            'repeated_videos': self.repeated_videos
        }

        pickle.dump(val_object, open(f'dataset_statistics_{str(time.time())}.pickle', 'wb'))


stats = DatasetStatistics()
