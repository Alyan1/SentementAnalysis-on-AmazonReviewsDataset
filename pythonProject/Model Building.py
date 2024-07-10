##  Loading Data    ##
import pandas as pd
from scipy.sparse import load_npz
tfidf_features = load_npz("tfidf_features.npz")
data = pd.read_csv("cleaned_train")
sentiment_labels = data["polarity"]

##  Splitting Data into Training and Testing Sets   ##
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, sentiment_labels, test_size=0.2, random_state=42)  # 80% training, 20% testing

##  Model Training  ##
# MultinomialNB ( Naive Bayes) is suitable for multi-class classification (positive or negative)
from sklearn.naive_bayes import MultinomialNB
sentiment_model = MultinomialNB()
sentiment_model.fit(X_train, y_train)

##  Model Evaluation    ##
from sklearn.metrics import accuracy_score, precision_score, recall_score

sentiment_model.fit(X_train, y_train)
predictions = sentiment_model.predict(X_test)
print('Accuracy = ',accuracy_score(y_test, predictions))

##  (Naive Bayes) Above code also working fine but this can also be used :
# y_pred = sentiment_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average="weighted")
# recall = recall_score(y_test, y_pred, average="weighted")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

## Saving Model ##
from joblib import dump
dump(sentiment_model, "sentiment_model.pkl")  # Save as a pickle file