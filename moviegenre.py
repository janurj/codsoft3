import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

def load_train_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    plot_summaries = []
    genres = []

    for line in data:
        parts = line.strip().split(' ::: ')
        if len(parts) == 4:
            _, _, genre, plot_summary = parts
            plot_summaries.append(plot_summary)
            genres.append(genre)
        else:
            print(f"Skipping line: {line.strip()}")
    
    if not plot_summaries:
        raise ValueError(f"No valid data found in the file: {file_path}")
    
    return pd.DataFrame({'plot_summary': plot_summaries, 'genre': genres})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load train data
train_file_path = r"C:\intern\archive (2)\Genre Classification Dataset\train_data.txt"
movie_data = load_train_data(train_file_path)
movie_data['plot_summary'] = movie_data['plot_summary'].apply(preprocess_text)

# Vectorize the data
vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(movie_data['plot_summary'])
y = movie_data['genre']

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(x_val)
print("\nValidation set classification report:")
print(classification_report(y_val, y_pred))

# Function to predict the genre based on user input
def predict_genre(movie_name, imdb_description):
    input_data = movie_name + " " + imdb_description
    input_data = preprocess_text(input_data)
    input_vector = vectorizer.transform([input_data])
    prediction = model.predict(input_vector)
    return prediction[0]

# Example of interactive input and prediction
movie_name = input("Enter the movie name: ")
imdb_description = input("Enter the IMDb description: ")
predicted_genre = predict_genre(movie_name, imdb_description)
print(f"\nPredicted Genre for '{movie_name}': {predicted_genre}")
