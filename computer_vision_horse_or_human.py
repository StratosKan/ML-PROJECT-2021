import PIL
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'horse-or-human/training/'

train_datagen = ImageDataGenerator(rescale=1/255)  # resize

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),  # tuple
    class_mode='binary'      # categorical
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),
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

RMSprop = tf.keras.optimizers.RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(   # requires pillow lib // if old version use model.fit_generator
    train_generator,
    epochs=15
)