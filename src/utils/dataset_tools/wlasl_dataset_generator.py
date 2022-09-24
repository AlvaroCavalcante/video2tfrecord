"""
This script is used to iterate over wlasl2000 dataset and
generate the splits for training, test and validation, besides
a csv file containing the video names and labels
"""
import shutil
import json
import pandas as pd


def generate_wlasl_dataset(dataset_json_content, new_dataset_path, current_dataset_path):
    classes = []
    videos = []
    split = []

    for class_n, sign in enumerate(dataset_json_content):
        classes.extend([class_n] * len(sign.get('instances')))
        for instance in sign.get('instances'):
            videos.append(instance.get('video_id'))
            split.append(instance.get('split'))

            video = f'{current_dataset_path}{instance.get("video_id")}.mp4'
            new_video = f'{new_dataset_path}wlasl_{instance.get("split")}/{instance.get("video_id")}.mp4'

            shutil.copyfile(video, new_video)

    df = pd.DataFrame(list(zip(videos, classes, split)),
                      columns=['video_name', 'label', 'split'])

    df[df['split'] == 'train'].to_csv(f'{new_dataset_path}wlasl_train/labels.csv', index=False)
    df[df['split'] == 'val'].to_csv(f'{new_dataset_path}wlasl_val/labels.csv', index=False)
    df[df['split'] == 'test'].to_csv(f'{new_dataset_path}wlasl_test/labels.csv', index=False)


if __name__ == '__main__':
    dataset_json_content = json.load(
        open('/home/alvaro/Desktop/WLASL/start_kit/WLASL_v0.3.json'))

    new_dataset_path = '/home/alvaro/Downloads/DATASETS/'
    current_dataset_path = '/home/alvaro/Downloads/DATASETS/WLASL2000/'

    generate_wlasl_dataset(dataset_json_content,
                           new_dataset_path, current_dataset_path)
