##   Loading train dataset  ##
import  pandas as pd
train=pd.read_csv('train.csv')
train.columns = ["polarity", "title", "body"]

# print(train.shape())
# print(train.info())
# print(train.head())
# print(train.columns)
# print(train['body'].value_counts())

##   Sampling   ##
train = train.sample(100000, random_state=123,replace=False)
print('Before balancing the records\n',train['polarity'].value_counts())

##   Balancing    ##
polarity_2_train = train[train['polarity'] == 2]
rows_to_drop = polarity_2_train.sample(n=344, random_state=123).index
train = train.drop(rows_to_drop)
print(train.shape)
print("After balancing the records\n", train["polarity"].value_counts())

##   Text Cleaning    ##
import re
from nltk.corpus import stopwords

# Lowercasing Text:
train['body'] = train['body'].str.lower()

# Removing HTML Tags and Special Characters:
def remove_tags(body):
    return re.sub(r"<.*?>", "", body)   # Removes everything between "<" and ">"
train["body"] = train["body"].apply(remove_tags)
def remove_special_chars(body):
  return re.sub(r"[^a-zA-Z0-9\s]", "", body)  # Removes characters except letters, numbers, and spaces

train["body"] = train["body"].apply(remove_special_chars)

#  Removing Punctuation:
def remove_punctuation(body):
    return re.sub(r"[^\w\s]", "", body) # Removes characters except letters, numbers, underscores, and spaces)

# Removing Stop Words:
stop_words = stopwords.words("english")
def remove_stopwords(body):
  return " ".join([word for word in body.split() if word not in stop_words])

train["body"] = train["body"].apply(remove_stopwords)

# Optional: Stemming or Lemmatization:
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stemming(body):
  return " ".join([stemmer.stem(word) for word in body.split()])
train["body"] = train["body"].apply(stemming)

# Saving the Cleaned Data:
train.to_csv("cleaned_train", index=False)  # Save to a new CSV file
