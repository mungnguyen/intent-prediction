import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def get_data(file_name):
    names = ["Intent", "Usersay"]
    data = pd.read_table(file_name, header=None, names=names)

    return data["Usersay"], data["Intent"]


def vectorize_data(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer(
                              tokenizer=nltk.word_tokenize, 
                              stop_words=stopwords.words()
                            )

    X_train = vectorizer.fit_transform(X_train).A
    X_test = vectorizer.transform(X_test).A

    """# Label
    """
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("Shape of X_train: {}".format(X_train.shape))
    print("Shape of X_test : {}".format(X_test.shape))

    print("Shape of y_train: {}".format(y_train.shape))
    print("Shape of y_test : {}".format(y_test.shape))

    print(f"Num of classes: {len(le.classes_)}")

    return X_train, y_train, X_test, y_test, vectorizer, le