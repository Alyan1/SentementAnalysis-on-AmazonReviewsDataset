import re
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB  # Example: Naive Bayes model

# Define functions for text cleaning (replace with your specific cleaning steps)
def remove_html_tags(body):
  return re.sub(r"<.*?>", "", body)

def remove_special_chars(body):
  return re.sub(r"[^a-zA-Z0-9\s]", "", body)

def remove_stopwords(body):
  stop_words = stopwords.words("english")
  return " ".join([word for word in body.split() if word not in stop_words])

def clean_text(body):
  body = body.lower()  # Lowercase
  body = remove_html_tags(body)
  body = remove_special_chars(body)
# Optionally uncomment the following lines for stop word removal and stemming/lemmatization
  body = remove_stopwords(body)
# Implement stemming or lemmatization (optional)
  return body

# Load TF-IDF vectorizer (assuming saved as "tfidf_vectorizer.pkl")
from joblib import load
tfidf_vectorizer = load("tfidf_vectorizer.pkl")

# Load sentiment analysis model (assuming saved as "sentiment_model.pkl")
sentiment_model = load("sentiment_model.pkl")

# Get new review text
new_review = input('Enter a review: ')
# Preprocess the new review
preprocessed_text = clean_text(new_review)

# Transform the preprocessed text into TF-IDF features
new_review_features = tfidf_vectorizer.transform([preprocessed_text])

# Predict sentiment for the new review
sentiment_prediction = sentiment_model.predict(new_review_features)

# Print the predicted sentiment

if sentiment_prediction[0] == 1:
    print("Predicted sentiment: Negative")
else:    print("Predicted sentiment: Positive")






