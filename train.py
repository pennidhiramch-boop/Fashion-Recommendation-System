import os
import pickle
import warnings

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

warnings.filterwarnings("ignore")


# =====================================================
# Project Paths
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
IMAGE_FOLDER = os.path.join(PROJECT_DIR, "images", "images")
MODEL_FOLDER = os.path.join(PROJECT_DIR, "models")
DATASET_PATH = os.path.join(
    DATA_FOLDER,
    "styles.csv"
)


FEATURE_FILE = os.path.join(
    MODEL_FOLDER,
    "feature_vectors.pkl"
)

FILENAME_FILE = os.path.join(
    MODEL_FOLDER,
    "filenames.pkl"
)

KNN_FILE = os.path.join(
    MODEL_FOLDER,
    "knn_model.pkl"
)

os.makedirs(MODEL_FOLDER, exist_ok=True)


# =====================================================
# Load ResNet50
# =====================================================

print("=" * 50)
print("Loading ResNet50 Model...")
print("=" * 50)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

model = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

print("Model Loaded Successfully!\n")
# =====================================================
# Feature Extraction
# =====================================================

def extract_feature(image_path):
    """
    Extract a 2048-dimensional feature vector
    from an image using ResNet50.
    """

    try:

        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(img_array)

        feature = model.predict(
            img_array,
            verbose=0
        ).flatten()

        feature = normalize(
            feature.reshape(1, -1)
        ).flatten()

        return feature

    except Exception as e:

        print(f"Error: {image_path}")
        print(e)

        return None


# =====================================================
# Dataset Loader
# =====================================================

print("=" * 50)
print("Loading Clean Dataset...")
print("=" * 50)

df = pd.read_csv(
    DATASET_PATH,
    on_bad_lines="skip"
)

# Keep only rows with a valid product ID
df = df.dropna(subset=["id"])
df["id"] = df["id"].astype(int)

print(f"Total Products : {len(df)}")
print(df.head())

print(f"Total Products : {len(df)}")


# =====================================================
# Extract Features
# =====================================================

feature_list = []
filename_list = []

print("\nExtracting Image Features...\n")

for product_id in tqdm(df["id"]):

    image_path = os.path.join(IMAGE_FOLDER, f"{product_id}.jpg") 
    if not os.path.isfile(image_path):
        continue

    feature = extract_feature(image_path)

    if feature is not None:
        feature_list.append(feature)
        filename_list.append(image_path)

print("\nFeature Extraction Completed!")

print(f"Total Features : {len(feature_list)}")
# =====================================================
# Train KNN Model
# =====================================================

print("\n" + "=" * 50)
print("Training Nearest Neighbors Model...")
print("=" * 50)

knn = NearestNeighbors(
    n_neighbors=6,
    algorithm="brute",
    metric="euclidean"
)

knn.fit(np.array(feature_list))

print("KNN Model Trained Successfully!")


# =====================================================
# Save Models
# =====================================================

print("\nSaving Files...")

with open(FEATURE_FILE, "wb") as f:
    pickle.dump(feature_list, f)

with open(FILENAME_FILE, "wb") as f:
    pickle.dump(filename_list, f)

with open(KNN_FILE, "wb") as f:
    pickle.dump(knn, f)

print("Feature vectors saved.")
print("Image filenames saved.")
print("KNN model saved.")


# =====================================================
# Summary
# =====================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Images Processed : {len(filename_list)}")
print(f"Feature Vectors  : {len(feature_list)}")
print(f"KNN Neighbors    : {knn.n_neighbors}")

print("\nSaved Files:")

print(FEATURE_FILE)
print(FILENAME_FILE)
print(KNN_FILE)

print("\nProject is ready for recommendation!")