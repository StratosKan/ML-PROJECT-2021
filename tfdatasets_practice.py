import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")
for item in mnist_data:
    print(item)

mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    # print(item['image'])
    print(item['label'])

mnist_test, info = tfds.load(name="fashion_mnist", with_info="true")
print(info)

# horse or human train with tfds


def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    return image, label


data = tfds.load("horses_or_humans", split="train", as_supervised=True)  # horses or humans dataset throws LOG ERROR
train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(10)  # batching issue
print(train_batches)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary classification - horse or human?
])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_batches, epochs=10)

# horse or human validation with tfds

val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
validation_batches = val_data.batch(32)

history = model.fit(train_batches, epochs=10, validation_data=validation_batches, validation_steps=1)
