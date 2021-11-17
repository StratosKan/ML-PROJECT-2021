import tensorflow as tf
import tensorflow_datasets as tfds

data, info = tfds.load("mnist", with_info=True)
print(info)

filename = "/root/tensorflow_datasets/mnist/3.0.0/mnist-test.tfrecord-00000-of-00001"
raw_dataset = tf.data.TFRecordDataset(filename)
# for raw_record in raw_dataset.take(1):
#    print(repr(raw_record))  # Return the canonical string representation of the object.

# Create a description of the features
feature_description = {
    'image': tf.io.FixedLenFeature([], dtype=tf.string),
    'label': tf.io.FixedLenFeature([], dtype=tf.int64),
}


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above
    return tf.io.parse_single_example(example_proto, feature_description)


parse_dataset = raw_dataset.map(_parse_function)
for parsed_record in parse_dataset.take(1):
    print(parsed_record)
