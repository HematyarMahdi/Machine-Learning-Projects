# spam_detector.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import string
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Load Dataset
print("📥 Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Step 2: Clean the Text
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    words = text.split()  # tokenize
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

df['clean_message'] = df['message'].apply(clean_text)

# Step 3: Convert Text to Features
print("🧠 Vectorizing text...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_message'])

# Step 4: Label Encoding
y = df['label'].map({'ham': 0, 'spam': 1})

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
print("🤖 Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
print("📊 Evaluating model...")
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict New Messages
def predict_spam(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Step 9: Test Custom Predictions
print("\n🔮 Testing custom predictions...")
test_messages = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "Hey, just wanted to check if we're still on for lunch tomorrow?",
    "Urgent: Your account has been compromised. Click the link to reset your password.",
    "See you at the meeting later today!"
]

for msg in test_messages:
    print(f"\n📧 Message: {msg}")
    print("Prediction:", predict_spam(msg))