import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

def read_tfrecord(example_proto):   
    face = []
    hand_1 = []
    hand_2 = []

    for image_count in range(24):
        face_stream = 'face' + '/' + str(image_count)
        hand_1_stream = 'hand_1' + '/' + str(image_count)
        hand_2_stream = 'hand_2' + '/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(example_proto, features=feature_dict)

        # width = tf.cast(features['width'], tf.int32)
        # height = tf.cast(features['height'], tf.int32)
        width = 50 
        height = 50

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)
        
        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)

        label = tf.cast(features['label'], tf.int32)

    return [face, hand_1, hand_2], label

def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    image = tf.cast(image, dtype='uint8')
    return image

def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset 


def prepare_for_training(ds, shuffle_buffer_size=5):
    ds.cache() # I can remove this to don't use cache or use cocodata.tfcache
    ds = ds.repeat()
    ds = ds.batch(4)

    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(4)
    ds = ds.prefetch(1)
    return ds

def load_data_tfrecord(tfrecord_path):
  dataset = load_dataset(tfrecord_path)

  dataset = prepare_for_training(dataset)
  return dataset

tf_record_path = tf.io.gfile.glob('/home/alvaro/Área de Trabalho/video2tfrecord/example/train/*.tfrecords')
row = 4; col = 4
#row = min(row,15//col)

all_elements = load_data_tfrecord(tf_record_path).unbatch()
augmented_element = all_elements.repeat().batch(1)


for (seq, label) in augmented_element:
    plt.figure(figsize=(15,int(15*row/col)))
    for j in range(row*col):
        plt.subplot(row,col,j+1)
        plt.axis('off')
        plt.imshow(np.array(seq[1, 2, j,]))
    plt.show()
