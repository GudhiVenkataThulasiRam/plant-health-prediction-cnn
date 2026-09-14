import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


# =========================================================
# 1. DATASET PATHS
# =========================================================

train_dir = "Train/Train"
validation_dir = "Validation/Validation"
test_dir = "Test/Test"


# =========================================================
# 2. IMAGE SETTINGS
# =========================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# =========================================================
# 3. LOAD DATASET
# =========================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Get class names
class_names = train_data.class_names

print("\nClasses:", class_names)


# =========================================================
# 4. CREATE CNN MODEL
# =========================================================

model = tf.keras.Sequential([

    # Normalize pixel values
    tf.keras.layers.Rescaling(
        1./255,
        input_shape=(128, 128, 3)
    ),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Convert feature maps to vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# =========================================================
# 5. COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 6. DISPLAY MODEL
# =========================================================

model.summary()


# =========================================================
# 7. TRAIN MODEL
# =========================================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5
)


# =========================================================
# 8. TRAINING & VALIDATION ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.close()


# =========================================================
# 9. TRAINING & VALIDATION LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.close()


# =========================================================
# 10. EVALUATE MODEL ON TEST DATA
# =========================================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# =========================================================
# 11. MAKE PREDICTIONS
# =========================================================

y_true = []
y_pred = []

for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# =========================================================
# 14. PLOT CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Plant Health Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)


# Add numbers inside matrix
for i in range(len(class_names)):
    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix saved successfully!")


# =========================================================
# 15. SAVE TRAINED MODEL
# =========================================================

model.save(
    "plant_health_cnn.keras"
)

print("Model saved successfully!")