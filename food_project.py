import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Food101 Dataset
# -----------------------------
dataset_name = "food101"

# Load dataset: 80% train, 20% test
(train_ds, test_ds), ds_info = tfds.load(
    dataset_name,
    split=['train[:80%]', 'train[80%:]'],  # First 80% = training, last 20% = testing
    as_supervised=True,                    # Gives us (image, label) tuples
    with_info=True                         # Loads metadata (class names, num classes)
)

print("Number of classes:", ds_info.features['label'].num_classes)
print("Class names sample:", ds_info.features['label'].names[:10])  # Show first 10 names

# -----------------------------
# Step 2: Visualize Sample Images
# -----------------------------
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(ds_info.features['label'].names[label])
    plt.axis("off")
plt.show()

# -----------------------------
# Step 3: Preprocess Data
# -----------------------------
IMG_SIZE = (128, 128)  # Can reduce to (64, 64) if slow
BATCH_SIZE = 32        # Reduce if you run into memory issues

def preprocess(image, label):
    """
    Resizes image and normalizes pixel values to [0, 1].
    - image: RGB TensorFlow tensor of shape (H, W, 3)
    - label: integer label for class
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

# Map preprocessing to each (image, label) pair, batch, and prefetch
train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Step 4: Build CNN Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),  # Helps reduce overfitting
    tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

# -----------------------------
# Step 5: Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# -----------------------------
# Step 6: Train Model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,  # Can increase later
    callbacks=[callback]
)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.2%}")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# Step 8: Make Predictions
# -----------------------------
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)

plt.figure(figsize=(12, 12))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(images[i].numpy())
    pred_class = ds_info.features['label'].names[predicted_labels[i]]
    true_class = ds_info.features['label'].names[labels[i]]
    plt.title(f"Pred: {pred_class}\nTrue: {true_class}",
              color=("green" if pred_class == true_class else "red"))
    plt.axis("off")
plt.show()