import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Training data shape:", train_images.shape)
print("Test data shape:", test_images.shape)

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0


# Show the first training image (before normalization)
plt.figure(figsize=(4,4))
plt.imshow(test_images[7512], cmap='gray')  # cmap='gray' = grayscale colormap
plt.title(f"Label: {test_labels[7512]}")
plt.colorbar()  # shows the value scale on the side
plt.show()

# Print pixel values from a small 5x5 top-left corner
print("First 5x5 pixel values of first image:\n", train_images[0][:5, :5])

model = keras.Sequential([
    # Initial Shape is (60000, 28, 28) for training images
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 into 784 values
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    keras.layers.Dense(64, activation='relu'),  # Another hidden layer with 64 neurons
    keras.layers.Dense(10, activation='softmax') # Output layer for 10 digits
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\nTest accuracy:", test_acc)

predictions = model.predict(test_images)
print("First prediction probabilities:", predictions[7512])
print("Predicted label:", np.argmax(predictions[7512]))
print("Actual label:", test_labels[7512])


