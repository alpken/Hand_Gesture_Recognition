import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: Image processing and manipulation using OpenCV
def load_images_from_folder(folder):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        label = class_folder
        class_folder_path = os.path.join(folder, class_folder)
        for filename in os.listdir(class_folder_path):
            img = cv2.imread(os.path.join(class_folder_path, filename))
            if img is not None:
                # Resizing images to 64x64 for CNN input
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load and process the images
image_folder = "path_to_image_folder"
X, y = load_images_from_folder(image_folder)

# Step 2: Data preparation
# Convert the images to grayscale (if needed) and normalize pixel values
X = X / 255.0

# Convert categorical labels to numerical (if not already)
label_map = {label: idx for idx, label in enumerate(np.unique(y))}
y_num = np.array([label_map[label] for label in y])

# One-hot encoding of labels for multi-class classification
y_one_hot = to_categorical(y_num)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Step 3: Building the CNN model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_map), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Training the CNN model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 5: Model evaluation
# Predicting the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Classification report
print(classification_report(y_true, y_pred_classes))

# Step 6: Visualizing the confusion matrix with heat map
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Step 7: Print F1 score (optional, already part of classification report)
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f'F1 Score: {f1}')
