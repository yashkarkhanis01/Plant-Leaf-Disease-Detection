import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up GPU configuration
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer for better feature extraction
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))  # Adding dropout to reduce overfitting
classifier.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generation and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'D:/Yash/GitHub/Plant Leaf Disease Detection/Dataset/train',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

valid_set = test_datagen.flow_from_directory(
    'D:/Yash/GitHub/Plant Leaf Disease Detection/Dataset/val',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

# Print class labels
labels = (training_set.class_indices)
print(labels)

# Calculate steps per epoch
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = valid_set.samples // valid_set.batch_size

# Fit the model
classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=valid_set,
    validation_steps=validation_steps
)

# Save the model
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save_weights("model_weights.weights.h5")
classifier.save("model.h5")
print("Saved model to disk")
