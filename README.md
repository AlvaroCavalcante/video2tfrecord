
# Description
This project was forked and based on the [video2tfrecord](https://github.com/ferreirafabio/video2tfrecord) library. The goal of this work is to help to transform a video dataset into a series of TFRecord files used on some ML frameworks like TensorFlow. This project was applied in the creation of a sign language recognition system, where the code here was used as the pre processing step to transform the raw data into different images that were feed to the model.


# Reading from tfrecord
To check the results (the images of the video stored in the tfrecord) run the ```read_dataset.py``` script.

# Parameters and storage details
By adjusting the parameters at the top of the code you can control:
- input dir (containing all the video files)
- output dir (to which the tfrecords should be saved)
- resolution of the images
- video file suffix (e.g. *.avi) as RegEx(!include asterisk!)
- number of frames per video that are actually stored in the tfrecord entries (can be smaller than the real number of frames)
- image color depth
- if optical flow should be added as a 4th channel
- number of videos a tfrecords file should contain


The videos are stored as features in the tfrecords. Every video instance contains the following data/information:
- feature[path] (as byte string while path being "blobs/i" with 0 <= i <=number of images per video)
- feature['height'] (while height being the image height, e.g. 128)
- feature['width'] (while width being the image width, e.g. 128)
- feature['depth'] (while depth being the image depth, e.g. 4 if optical flow used)
