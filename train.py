"""
Standalone version of face_recognition.ipynb — trains the MobileNetV2
FER-2013 emotion classifier and saves models/emotion_recognition_mobilenetv2.keras
for the FastAPI service to load.

Usage:
    python train.py

Needs fer2013.csv in the working directory (see notebook for the download step).

CHANGES vs original version (to fix low accuracy ~33%):
  - Removed hardcoded steps_per_epoch/validation_steps so each epoch actually
    sees the FULL training set, not just a small slice of it.
  - Increased epochs (20 for feature-extraction phase, 15 for fine-tuning)
    with early stopping so training can converge properly.
  - Added class_weight so the model doesn't ignore underrepresented classes
    like 'disgust' (FER-2013 is heavily imbalanced).
  - Unfroze more of the base model (last 50 layers instead of 30) in phase 2
    so fine-tuning has a bigger effect.
"""

import os

import gdown
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

FILE_ID = "1V0ELMWF5qSmCF4rOuKChnizgp-w49I5a"  # public fer2013.csv mirror
CSV_PATH = "fer2013.csv"
IMG_SIZE = 96
BATCH_SIZE = 64
TRAIN_DIR = "fer2013_data/train"
TEST_DIR = "fer2013_data/test"
os.makedirs("models", exist_ok=True)


def prepare_dataset():
    if not os.path.exists(CSV_PATH):
        print("Downloading fer2013.csv ...")
        gdown.download(id=FILE_ID, output=CSV_PATH, quiet=False)

    # Skip re-splitting into folders if it was already done in a previous run.
    if os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR) and any(os.scandir(TRAIN_DIR)):
        print("fer2013_data/ already prepared, skipping re-split.")
        return

    df = pd.read_csv(CSV_PATH)
    emotion_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
    label_names = [emotion_map[i] for i in range(7)]

    for split_name in ["train", "test"]:
        for cls in label_names:
            os.makedirs(f"fer2013_data/{split_name}/{cls}", exist_ok=True)

    def usage_to_split(u):
        return "train" if u == "Training" else "test"

    counts = {"train": 0, "test": 0}
    for _, row in df.iterrows():
        split_name = usage_to_split(row["Usage"])
        label = emotion_map[int(row["emotion"])]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels, mode="L").convert("RGB")
        img.save(f"fer2013_data/{split_name}/{label}/{counts[split_name]}.jpg")
        counts[split_name] += 1
    print("Dataset ready:", counts)


def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base_model


def main():
    prepare_dataset()

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.1,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation", shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb",
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
    )
    class_names = list(train_gen.class_indices.keys())
    print("Classes:", class_names)

    # FER-2013 is heavily imbalanced (e.g. 'disgust' has ~15x fewer images than
    # 'happy'). Without class weights the model mostly ignores rare classes.
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight = dict(enumerate(class_weights_array))
    print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight.items()})

    model, base_model = build_model(len(class_names))

    # Phase 1: feature extraction (base frozen). Let each epoch see the FULL
    # training set — no steps_per_epoch override — and give it enough epochs
    # (with early stopping) to actually converge.
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ModelCheckpoint("models/best_model_phase1.keras", monitor="val_accuracy", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ],
    )

    # Phase 2: fine-tune more of the base model's top layers, with a low LR
    # and enough epochs to meaningfully adapt the ImageNet features to faces.
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ModelCheckpoint("models/best_model_finetuned.keras", monitor="val_accuracy", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ],
    )

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    model.save("models/emotion_recognition_mobilenetv2.keras")
    print("Saved models/emotion_recognition_mobilenetv2.keras")


if __name__ == "__main__":
    main()