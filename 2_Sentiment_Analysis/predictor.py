# predictor.py
from text_preprocessor import clean_text
from model import load_model

def predict_sentiment(text):
    model, vectorizer = load_model()
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"
