"""Convert robocup dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""



import tensorflow as tf
import os
import pandas as pd
import yaml

# It is required to run beforehand
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
from object_detection.utils import dataset_util


flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '',
                       'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations CSV file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations CSV file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

tf.flags.DEFINE_string('yml_filename', '', 'Classes yaml file.')
FLAGS = flags.FLAGS


def create_tf_record_from_csv(annotations_file, image_dir, yml_filename, output_path, num_shards):

    # Load csv file
    df = pd.read_csv(annotations_file, sep=',')

    # Load yaml file
    with open(yml_filename, 'r') as yml_file:
        classes_dict = yaml.load(yml_file, Loader=yaml.FullLoader)

    # Iterate over the annotations_file
    for _, row in df.iterrows():

        image_name = row['image_name']
        xmin =
        print(row['image_name'], row['xmin'], row['xmax'], row['ymin'], row['ymax'], row['class_id'], classes_dict[row['class_id']])

    height = None # Image height
    # width = None # Image width
    # filename = None # Filename of the image. Empty if image is not from file
    # encoded_image_data = None # Encoded image bytes
    # image_format = None # b'jpeg' or b'png'
    #
    # xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    # xmaxs = [] # List of normalized right x coordinates in bounding box
    #            # (1 per box)
    # ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    # ymaxs = [] # List of normalized bottom y coordinates in bounding box
    #            # (1 per box)
    # classes_text = [] # List of string class name of bounding box (1 per box)
    # classes = [] # List of integer class id of bounding box (1 per box)
    #
    # tf_example = tf.train.Example(features=tf.train.Features(feature={
    #     'image/height': dataset_util.int64_feature(height),
    #     'image/width': dataset_util.int64_feature(width),
    #     'image/filename': dataset_util.bytes_feature(filename),
    #     'image/source_id': dataset_util.bytes_feature(filename),
    #     'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    #     'image/format': dataset_util.bytes_feature(image_format),
    #     'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    #     'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    #     'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    #     'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    #     'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    #     'image/object/class/label': dataset_util.int64_list_feature(classes),
    # }))
    # return tf_example


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'
    assert FLAGS.yml_filename, '`yml_filename` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_output_path = os.path.join(FLAGS.output_dir, 'robocup_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'robocup_val.record')

    create_tf_record_from_csv(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        FLAGS.yml_filename,
        train_output_path,
        num_shards=1)

    # create_tf_record_from_csv(
    #     FLAGS.val_annotations_file,
    #     FLAGS.val_image_dir,
    #     val_output_path,
    #     FLAGS.include_masks,
    #     num_shards=1)


if __name__ == '__main__':
    tf.app.run()
