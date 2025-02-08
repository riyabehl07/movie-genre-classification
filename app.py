import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Define file paths
BASE_DIR = r"C:/Users/Riya behl/Desktop/movie-genre/Genre Classification Dataset"
train_file = os.path.join(BASE_DIR, "train_data.txt")
test_file = os.path.join(BASE_DIR, "test_data.txt")
output_file = os.path.join(BASE_DIR, "predictions.txt")


def load_train_dataset(file_path):
    """Load training data with ID, Title, Genre, and Description."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: {file_path} not found!")

    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if len(parts) == 4:  # Ensure proper format
                texts.append(parts[3])  # Description
                labels.append(parts[2])  # Genre
            else:
                print(f"⚠️ Skipping invalid line: {line.strip()}")  # Debugging print

    if not texts:
        raise ValueError(f"❌ Error: No valid data found in {file_path}")

    return texts, labels


def load_test_dataset(file_path):
    """Load test data with ID, Title, and Description (no Genre)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: {file_path} not found!")

    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(":::")
            if len(parts) == 3:  # Ensure proper format
                texts.append(parts[2])  # Description
            else:
                print(f"⚠️ Skipping invalid line: {line.strip()}")  # Debugging print

    if not texts:
        raise ValueError(f"❌ Error: No valid data found in {file_path}")

    return texts


# Load datasets
X_train, y_train = load_train_dataset(train_file)
X_test = load_test_dataset(test_file)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r'\b\w+\b', min_df=1)

# Convert text data to numerical vectors
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict genres for test data
predictions = clf.predict(X_test_tfidf)

# Save predictions
with open(output_file, "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")

print(f"✅ Predictions saved to {output_file}")
