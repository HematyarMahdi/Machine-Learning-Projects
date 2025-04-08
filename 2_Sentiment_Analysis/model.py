# model.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_model(df, clean_column='clean_review'):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[clean_column])
    y = df['label'].map({'pos': 1, 'neg': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer, X_test, y_test, y

def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer
