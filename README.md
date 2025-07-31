# Hand_Gesture_Recognition
### Hand Gesture Recognition Using Convolutional Neural Networks (CNNs)


This project was completed as part of a course assignment to build a deep learning-based Hand Gesture Recognition System using Convolutional Neural Networks (CNNs). The goal of the project is to classify hand gesture images into distinct categories based on visual features learned directly from pixel data.

The workflow starts with image acquisition and preprocessing using OpenCV, where hand gesture images are loaded from a folder-based structure, resized to 64×64 pixels, and normalized. The dataset is automatically labeled based on folder names representing gesture classes. These labels are numerically encoded and converted to one-hot vectors for multi-class classification.

A CNN model was built using TensorFlow/Keras, comprising three convolutional layers for feature extraction, followed by fully connected layers for classification. Techniques such as dropout were used to reduce overfitting and improve generalization. The model was trained using the Adam optimizer with categorical crossentropy as the loss function.

After training, the model's performance was evaluated on unseen test data using metrics such as accuracy, F1-score, and a confusion matrix, which was visualized using a heatmap.

This project demonstrates the power of deep learning in recognizing and classifying hand gestures, forming the basis for applications in human-computer interaction, sign language translation, and gesture-based control systems.
