# Movie Genre Classification

## 📌 Project Overview
This project classifies movie descriptions into genres using Natural Language Processing (NLP) and Machine Learning. It utilizes the **TF-IDF** vectorization technique and a **Multinomial Naïve Bayes (or Logistic Regression)** model for classification.

## 📂 Dataset Structure
The dataset is sourced from:
> ftp://ftp.fu-berlin.de/pub/misc/movies/database/

## 📂 Dataset
https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

### 🔹 **Train Data (`train_data.txt`)**
```
ID ::: TITLE ::: GENRE ::: DESCRIPTION
ID ::: TITLE ::: GENRE ::: DESCRIPTION
...
```
- **ID**: Unique identifier for the movie.
- **TITLE**: Name of the movie.
- **GENRE**: The movie's genre (e.g., Action, Comedy, Drama).
- **DESCRIPTION**: A short summary of the movie.

### 🔹 **Test Data (`test_data.txt`)**
```
ID ::: TITLE ::: DESCRIPTION
ID ::: TITLE ::: DESCRIPTION
...
```
- **ID**: Unique identifier.
- **TITLE**: Movie title.
- **DESCRIPTION**: Movie description (without genre label).

### 🔹 **Solution File (`test_data_solution.txt`)**
Contains the correct genres for the test data, used for calculating accuracy.

## 🚀 Installation & Setup
### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo/movie-genre-classification.git
cd movie-genre-classification
```

### 2️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```
Ensure `nltk` stopwords are downloaded:
```python
import nltk
nltk.download('stopwords')
```

## 🏗️ Project Structure
```
📂 movie-genre-classification/
│-- app.py                     # Main script for training & predicting
│-- README.md                   # Project documentation
│-- Genre Classification Dataset/
│   ├── train_data.txt         # Training dataset
│   ├── test_data.txt          # Test dataset
│   ├── test_data_solution.txt # Ground truth for test data
│   ├── predictions.txt        # Model predictions
```

## 🛠️ Usage
### **1️⃣ Train & Test the Model**
Run the script:
```sh
python app.py
```

### **2️⃣ View Predictions**
Predictions are saved in:
```
Genre Classification Dataset/predictions.txt
```

## ⚙️ Features & Improvements
✅ **TF-IDF Vectorization**: Converts text data into numerical form.
✅ **Naïve Bayes / Logistic Regression Classifier**: Trained on movie descriptions.
✅ **Stopwords Removal**: Filters out common words.
✅ **Handles Invalid Data**: Skips incorrectly formatted lines.
✅ **Optimized for Speed**: Uses `max_features=10000` in TF-IDF.

## 🏆 Future Enhancements
🚀 **Deep Learning (LSTM/Transformer models)** for better predictions.
🚀 **More Features**: Use metadata like actors, director, and release year.
🚀 **Web API**: Deploy the model as an API for real-time classification.



