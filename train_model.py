import os
import pandas as pd
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_SLUG = "omkargurav/face-mask-dataset"
MODEL_FILENAME = "face_mask_model.h5"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 2
STEPS_PER_EPOCH = 120
VALIDATION_STEPS = 35

print("Downloading or locating dataset...")
dataset_path = kagglehub.dataset_download(DATASET_SLUG)
base_path = os.path.join(dataset_path, "data")
print("Dataset path:", dataset_path)
print("Classes:", os.listdir(base_path))

rows = []
for label in os.listdir(base_path):
    class_dir = os.path.join(base_path, label)
    if not os.path.isdir(class_dir):
        continue
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            rows.append([os.path.join(class_dir, filename), label])

if not rows:
    raise RuntimeError(f"No image files found in {base_path}")

print(f"Found {len(rows)} image files")
df = pd.DataFrame(rows, columns=["image_path", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
test_size = len(df) - train_size - val_size
print("Split sizes -> train:", train_size, "val:", val_size, "test:", test_size)

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_generator = val_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Building model...")
model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_FILENAME,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

print("Training model...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stop, checkpoint],
    verbose=1,
)

print("Evaluating on test set...")
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test accuracy: {accuracy:.4f}")

print(f"Saving model to {MODEL_FILENAME}...")
model.save(MODEL_FILENAME)
print("Model saved successfully.")
