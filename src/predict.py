import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "complaint_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("AI CUSTOMER COMPLAINT CLASSIFIER")
print("=" * 60)

while True:
    text = input("\nEnter customer complaint (or type exit): ")

    if text.lower() == "exit":
        break

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities)

    print("\nCategory :", prediction)
    print("Confidence:", f"{confidence:.2%}")