# Fashion Recommendation System

This is a simple fashion recommendation system that suggests similar clothing items based on an uploaded image.

I built this project to learn how image feature extraction and recommendation systems work using deep learning.

## Features

- Upload an image of a clothing item
- Extract image features using ResNet50
- Find visually similar products
- Display recommendations in a simple Streamlit web app

## Tech Stack

- Python
- TensorFlow
- ResNet50
- Scikit-learn
- Streamlit
- Pandas
- NumPy

## Project Structure

```
Fashion-Recommendation-System/
│
├── app.py
├── train.py
├── recommend.py
├── preprocess.py
├── requirements.txt
├── README.md
├── data/
└── models/
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/pennidhiramch-boop/Fashion-Recommendation-System.git
```

Go to the project folder.

```bash
cd Fashion-Recommendation-System
```

Install the required libraries.

```bash
pip install -r requirements.txt
```

Run the application.

```bash
streamlit run app.py
```

## Dataset

This project uses the Fashion Product Images dataset.

The image dataset is not included in this repository because it is too large for GitHub.

## What I Learned

While building this project, I learned:

- Image preprocessing
- Feature extraction using ResNet50
- K-Nearest Neighbors for recommendations
- Building a Streamlit application
- Using Git and GitHub for version control

## Future Improvements

- Better recommendation accuracy
- Filter by category and gender
- Deploy the project online
- Improve the UI

## Author

Pennidhi
