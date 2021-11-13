import tensorflow as tf


data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)  # 4th param is color channel (greyscale=1,colored=3)
training_images = training_images / 255.0                    # normalization
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 64 filters, 26x26
    tf.keras.layers.MaxPooling2D(2, 2),                                              # 13x13
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),                           # 11x11, 64x(64x9)+64
    tf.keras.layers.MaxPooling2D(2, 2),                                              # 5x5
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),                               # (5x5x64)x128+128(bias)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)                              # 128x10+10
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=50)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

# model.summary()
