import tensorflow_datasets as tfds
import tensorflow as tf
import multiprocessing

# part 1
train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)
file_pattern = f'C:/Users/PC/tensorflow_datasets/cats_vs_dogs/2.0.1/cats_vs_dogs-train.tfrecord*'
files = tf.data.Dataset.list_files(file_pattern)
print(len(files))
train_dataset = files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=4,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# part 2
def read_tfrecord(serialized_example):
    feature_description = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1)
    }
    example = tf.io.parse_single_example(
        serialized_example, feature_description
    )
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (300, 300))
    return image, example['label']


cores = multiprocessing.cpu_count()  # This number is not equivalent to the number of CPUs the current process can use.
# The number of usable CPUs can be obtained with len(os.sched_getaffinity(0))
print("Found " + cores + " CPU cores")
train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# ready to train
# model.fit(train_dataset, epochs=10, verbose=1)
