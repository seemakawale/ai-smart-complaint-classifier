import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "complaint_classifier.pkl")

data = {
    "text": [
        "I was charged twice for my order",
        "My payment failed but money was deducted",
        "I have not received my refund",
        "The transaction shows pending for hours",
        "My card payment was declined",
        "I cannot log into my account",
        "I forgot my password",
        "My account has been locked",
        "I cannot change my email address",
        "Someone changed my account details",
        "My order has not arrived",
        "The package is delayed",
        "I received the wrong product",
        "My product arrived damaged",
        "I want to cancel my order",
        "The delivery person never came",
        "I want to return my purchase",
        "How can I track my order",
        "The app keeps crashing",
        "The website is not working",
        "I cannot upload a document",
        "The application is extremely slow",
        "The mobile app shows an error",
        "Customer support is not responding",
        "I need help with my payment",
        "My refund is missing",
        "My account password does not work",
        "My order is late",
        "The product I received is damaged",
        "The application keeps crashing",
    ],
    "category": [
        "Payment",
        "Payment",
        "Payment",
        "Payment",
        "Payment",
        "Account",
        "Account",
        "Account",
        "Account",
        "Account",
        "Delivery",
        "Delivery",
        "Delivery",
        "Delivery",
        "Delivery",
        "Delivery",
        "Delivery",
        "Delivery",
        "Technical",
        "Technical",
        "Technical",
        "Technical",
        "Technical",
        "Technical",
        "Payment",
        "Payment",
        "Account",
        "Delivery",
        "Delivery",
        "Technical",
    ]
}

df = pd.DataFrame(data)

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df.to_csv(DATA_PATH, index=False)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("=" * 60)
print("AI CUSTOMER COMPLAINT CLASSIFIER")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, prediction, zero_division=0))

joblib.dump(model, MODEL_PATH)

print("\nModel saved:")
print(MODEL_PATH)

print("\nDataset saved:")
print(DATA_PATH)

print("\nTRAINING COMPLETE!")