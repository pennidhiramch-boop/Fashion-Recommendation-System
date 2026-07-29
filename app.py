import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize
from PIL import Image
import os

st.set_page_config(page_title="Fashion Recommendation System")

st.title("👕 Fashion Recommendation System")

model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

feature_list = pickle.load(open("models/feature_vectors.pkl", "rb"))
filenames = pickle.load(open("models/filenames.pkl", "rb"))
knn = pickle.load(open("models/knn_model.pkl", "rb"))

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

def extract_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = model.predict(img_array, verbose=0).flatten()
    feature = normalize(feature.reshape(1, -1)).flatten()
    return feature

if uploaded_file is not None:

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", width=250)

    feature = extract_feature(temp_path)

    distances, indices = knn.kneighbors([feature])

    st.subheader("Recommended Products")

    cols = st.columns(5)

    for i, col in enumerate(cols):
        img = Image.open(filenames[indices[0][i]])
        col.image(img, use_container_width=True)