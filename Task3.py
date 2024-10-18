import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Load and Preprocess Data

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize the image pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the image data to include the channel dimension
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Convert labels to categorical format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define Data Augmentation

# Create an ImageDataGenerator instance for real-time data augmentation
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1)

# Fit the data generator on the training images
datagen.fit(train_images)

# Build the Convolutional Neural Network (CNN)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model with Early Stopping

# Set up early stopping to monitor validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the data generator
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=20,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping])

# Evaluate the Model

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Visualize Training History

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Recognize Handwritten Characters

def predict_and_display(image):
    # Ensure the input image is in the correct shape
    if image.shape != (28, 28, 1):
        raise ValueError("Input image must have the shape (28, 28, 1)")

    # Preprocess the image for prediction
    image = np.array(image).reshape((1, 28, 28, 1)) / 255.0

    # Predict the label of the image
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Display the image along with the predicted label
    plt.imshow(image.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f'Predicted Label: {predicted_label}')
    plt.axis('off')
    plt.show()

    return predicted_label

# Sample predictions
sample_indices = [10, 15, 20]
for index in sample_indices:
    predicted_label = predict_and_display(test_images[index])
    print(f'Actual Label: {np.argmax(test_labels[index])}')
