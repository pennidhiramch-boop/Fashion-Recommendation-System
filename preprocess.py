import os
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
IMAGE_FOLDER = os.path.join(PROJECT_DIR, "images")

RAW_DATASET = os.path.join(DATA_FOLDER, "styles.csv")
CLEAN_DATASET = os.path.join(DATA_FOLDER, "cleaned_styles.csv")


# -----------------------------
# Helper Functions
# -----------------------------

def load_dataset(file_path):
    """Load the dataset."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print("Loading dataset...")

    df = pd.read_csv(file_path, on_bad_lines="skip")

    print("Dataset loaded successfully!")

    return df


def remove_missing_values(df):
    print("Removing missing values...")
    df = df.dropna()
    print(f"Remaining rows: {len(df)}")
    return df


def remove_duplicate_rows(df):
    print("Removing duplicate rows...")
    df = df.drop_duplicates()
    print(f"Remaining rows: {len(df)}")
    return df


# -----------------------------
# Image Validation
# -----------------------------

def keep_products_with_images(df):

    print("Checking product images...")

    image_exists = []

    for product_id in df["id"]:

        image_path = os.path.join(
            IMAGE_FOLDER,
            f"{product_id}.jpg"
        )

        image_exists.append(os.path.exists(image_path))

    df["image_exists"] = image_exists

    df = df[df["image_exists"] == True]

    print(f"Products with images: {len(df)}")

    return df


# -----------------------------
# Save Clean Dataset
# -----------------------------

def save_dataset(df):

    df.to_csv(
        CLEAN_DATASET,
        index=False
    )

    print("Clean dataset saved successfully!")
    print(CLEAN_DATASET)


# -----------------------------
# Main Function
# -----------------------------

def main():

    df = load_dataset(RAW_DATASET)

    print(f"\nOriginal Dataset Shape: {df.shape}")

    df = remove_missing_values(df)

    df = remove_duplicate_rows(df)

    df = keep_products_with_images(df)

    save_dataset(df)

    print("\nPreprocessing Completed Successfully!")


if __name__ == "__main__":
    main()