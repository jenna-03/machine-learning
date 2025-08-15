import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


dataset_name = "tf_flowers"  # We'll use flowers dataset here for speed — we can replace with fruits later
(train_ds, test_ds), ds_info = tfds.load(
    dataset_name,
    split=['train[:80%]', 'train[80%:]'],  # 80% training, 20% testing
    as_supervised=True,  # gives (image, label) pairs
    with_info=True       # so we can get metadata (like label names)
)

print("Number of classes:", ds_info.features['label'].num_classes)
print("Class names:", ds_info.features['label'].names)

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):  # Take first 9 samples
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(ds_info.features['label'].names[label])
    plt.axis("off")
plt.show()

IMG_SIZE = (128, 128)  # standard size for flowers; adjust later for Fruits360
BATCH_SIZE = 32        # number of images processed together

def preprocess(image, label):
    # Size of original image looks like (height, width, channels=3 for RGB)
    image = tf.image.resize(image, IMG_SIZE)   # Resize to 128x128
    image = image / 255.0                      # Normalize to 0–1
    return image, label

# Map is applying the preprocess function to each (image, label) pair by doing a for loop internally
# Batch groups multiple samples together for efficiency
# Prefetch overlaps data preprocessing and model execution while training
train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Add more layers for deeper networks if needed
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the 3D output to 1D for the dense layers
    # Takes shape of (batch_size, height, width, channels) and flattens it to (batch_size, height*width*channels)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer with number of classes equal to number of labels in dataset
    # Activation function softmax for multi-class classification
    tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=[callback]
)

# -----------------------------
# Step 9: Evaluate the model
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# Step 10: Make Predictions
# -----------------------------
# Get one batch of test data
for images, labels in test_ds.take(1):
    predictions = model.predict(images)  # shape = (batch_size, num_classes)
    predicted_labels = tf.argmax(predictions, axis=1)  # get class index
    true_labels = labels

# -----------------------------
# Step 11: Visualize Predictions
# -----------------------------
plt.figure(figsize=(12, 12))
for i in range(9):  # show first 9 images from the batch
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(images[i].numpy())  # convert tensor to numpy
    pred_class = ds_info.features['label'].names[predicted_labels[i]]
    true_class = ds_info.features['label'].names[true_labels[i]]
    plt.title(f"Pred: {pred_class}\nTrue: {true_class}",
              color=("green" if pred_class == true_class else "red"))
    plt.axis("off")

plt.show()

