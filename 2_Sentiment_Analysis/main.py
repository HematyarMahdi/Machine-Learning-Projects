# main.py
from data_loader import load_movie_data
from text_preprocessor import clean_text
from model import train_model
from predictor import predict_sentiment
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("📥 Loading data...")
    df = load_movie_data()

    print("🧹 Cleaning reviews...")
    df['clean_review'] = df['review'].apply(clean_text)

    print("🧠 Training model...")
    model, vectorizer, X_test, y_test, y = train_model(df)

    print("✅ Evaluating...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("\n🔮 Try a custom prediction:")
    test_1 = "This movie was a masterpiece! Truly touching."
    test_2 = "Worst film ever. Don’t waste your time."
    print(f"> '{test_1}' → {predict_sentiment(test_1)}")
    print(f"> '{test_2}' → {predict_sentiment(test_2)}")

if __name__ == "__main__":
    main()
