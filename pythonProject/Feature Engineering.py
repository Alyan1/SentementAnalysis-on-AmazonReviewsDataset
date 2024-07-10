##      Feature Engineering     ##
import pandas as pd
cleaned_train=pd.read_csv('cleaned_train')
# print(cleaned_train.shape)
# print(cleaned_train['polarity'].info())
# print(cleaned_train['polarity'].value_counts())

##  TF-IDF Implementation  ##
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
# Fit the vectorizer to the data (learns vocabulary and IDF weights)
tfidf_features = tfidf_vectorizer.fit_transform(cleaned_train["body"])
# Print the TF-IDF feature matrix shape
print(tfidf_features.shape)  # This will show the number of reviews (rows) and features (columns)
# print(tfidf_features)

## Saving the features ##
import spicy
from scipy.sparse import csr_matrix
# Assuming you have the `tfidf_features` matrix
sparse_tfidf_features = csr_matrix(tfidf_features)
spicy.sparse.save_npz("tfidf_features", sparse_tfidf_features)
from joblib import dump
dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

