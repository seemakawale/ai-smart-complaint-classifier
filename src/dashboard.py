import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "complaint_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="AI Customer Complaint Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Customer Complaint Classifier")

st.write(
    "Enter a customer complaint and the machine-learning model "
    "will classify the problem category."
)

text = st.text_area(
    "Customer Complaint",
    placeholder="Example: I was charged twice for my order..."
)

if st.button("Analyze Complaint"):

    if not text.strip():
        st.warning("Please enter a complaint.")
    else:

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]

        confidence = max(probabilities)

        st.success("Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Category", prediction)

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

        if confidence >= 0.80:
            priority = "HIGH"
        elif confidence >= 0.60:
            priority = "MEDIUM"
        else:
            priority = "REVIEW"

        st.metric("Classification Confidence", f"{confidence:.1%}")

        st.write("### Suggested Priority")

        if priority == "HIGH":
            st.error(priority)
        elif priority == "MEDIUM":
            st.warning(priority)
        else:
            st.info(priority)