import os
import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
DATASET_PATH = 'NITROHCSv1.0'
IMAGE_SIZE = (81, 81)
NUM_CLASSES = 47
SAVE_MODEL = True  # Set to False if you don't want to save the trained model
TRAIN_MODEL = False  # Set to False to use the saved model without retraining
MODEL_NAME = 'handwritten_model.h5'


# Load Data
def load_data():
    images, labels = [], []
    class_names = sorted(os.listdir(DATASET_PATH))  # Ensure class order is consistent
    class_mapping = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        class_path = os.path.join(DATASET_PATH, cls_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                img = cv2.resize(img, IMAGE_SIZE)  # Ensure size consistency
                images.append(img)
                labels.append(class_mapping[cls_name])

    images = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1) / 255.0  # Normalize
    labels = to_categorical(np.array(labels), NUM_CLASSES)

    return images, labels, class_names

# Define CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(81, 81, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and Save Model
def train_and_save():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model()
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(X_train)
    model.fit(X_train, y_train, epochs=17, batch_size=32, validation_data=(X_test, y_test))
    
    if SAVE_MODEL:
        model.save(MODEL_NAME)
        print(f'Model saved as {MODEL_NAME}')
        
    return model, class_names

# Load or Train Model
def get_model():
    if TRAIN_MODEL or not os.path.exists(MODEL_NAME):
        print("Training model...")
        model, class_names = train_and_save()
    else:
        print("Loading saved model...")
        model = load_model(MODEL_NAME)
        _, _, class_names = load_data()
    
    return model, class_names

# Predict Image Class
def predict_image(img_path):
    model, class_names = get_model()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.array(img).reshape(1, 81, 81, 1) / 255.0
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    plt.imshow(img.reshape(81, 81), cmap='gray')
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()
    return predicted_class

# Run Training or Load Model
if __name__ == '__main__':
    model, class_names = get_model()
    predict_image("")
    