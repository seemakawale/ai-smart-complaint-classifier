# AI Customer Complaint Classifier

A machine-learning NLP system that automatically classifies customer complaints into categories such as Payment, Account, Delivery, and Technical Support.

## Problem

Customer-support teams receive large numbers of complaints every day. Manually routing every complaint can be slow and inconsistent.

This project demonstrates how NLP and machine learning can automatically classify incoming complaints.

## Pipeline

Customer Complaint
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Logistic Regression
↓
Complaint Category
↓
Confidence Score
↓
Streamlit Dashboard

## Categories

- Payment
- Account
- Delivery
- Technical

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib



## Run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run src/dashboard.py

