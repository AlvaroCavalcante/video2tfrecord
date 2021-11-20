import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

def read_tfrecord(example_proto):   
    image_seq = []

    for image_count in range(24-8):
        path = 'blob' + '/' + str(image_count)

        feature_dict = {path: tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)}

        features = tf.io.parse_single_example(example_proto, features=feature_dict)

        image = tf.image.decode_jpeg(features[path], channels=3)
        image = tf.reshape(image, tf.stack([HEIGHT, WIDTH, 3]))
        image = tf.reshape(image, [1, HEIGHT, WIDTH, 3])
        image_seq.append(image)

    image_seq = tf.concat(image_seq, 0)

    return image_seq

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

tf_record_path = "/home/alvaro/Documentos/video2tfrecord/example/output/batch_1_of_235.tfrecords"
WIDTH = 800
HEIGHT = 600

row = 4; col = 4
#row = min(row,15//col)

all_elements = load_data_tfrecord(tf_record_path).unbatch()
augmented_element = all_elements.repeat().batch(1)

for seq in augmented_element:
    plt.figure(figsize=(15,int(15*row/col)))
    for j in range(row*col):
        plt.subplot(row,col,j+1)
        plt.axis('off')
        plt.imshow(np.array(seq[0, j,]))
    plt.show()
