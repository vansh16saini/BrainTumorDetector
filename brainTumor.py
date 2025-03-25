import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Define ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increase rotations
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Add vertical flip
    brightness_range=[0.8, 1.2],  # Adjust brightness
    validation_split=0.2
)


# Load dataset
dataset_path = r"D:\GitHUb\BrainTumorDetector\brain_tumor_dataset"

train_generator = train_datagen.flow_from_directory(
    dataset_path,  
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,  
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

# Display some sample images
def visualize_images():
    x_batch, y_batch = next(train_generator)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_batch[i])
        ax.set_title("Tumor" if y_batch[i] == 1 else "No Tumor")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_images()
    print("Dataset Loaded Successfully!")
#-----------------------------------------------------------------------------
# Load necessary modules
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW

# Load Pretrained VGG16 Model (without the top layer)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze convolutional layers (fine-tune later)

# Modify the Model for Binary Classification
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)  # Increased neurons
x = BatchNormalization()(x)  # Normalize activations
x = Dropout(0.5)(x)  # Regularization
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)  # Binary classification (Tumor / No Tumor)

model = Model(inputs=base_model.input, outputs=x)

# Compile the Model with a tuned learning rate
model.compile(optimizer=AdamW(learning_rate=0.00005), loss="binary_crossentropy", metrics=["accuracy"])

# Train the Model for more epochs
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,  # Increased from 10 to 25
    batch_size=32  # Ensure a good batch size
)


# 6️⃣ Save the Model
model.save("brain_tumor_model.h5")
print("Model saved successfully!")
