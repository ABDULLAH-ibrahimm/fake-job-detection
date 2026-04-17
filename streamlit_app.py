import os
import joblib
import streamlit as st
from preprocess import clean_text

MODEL_PATH = "models/best_model.pkl"
MODEL_NAME_PATH = "models/best_model_name.txt"

st.set_page_config(page_title="Fake Job Detection", page_icon="🔎", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

def load_model_name():
    if os.path.exists(MODEL_NAME_PATH):
        with open(MODEL_NAME_PATH, "r") as f:
            return f.read().strip()
    return "Unknown"

model = load_model()
best_model_name = load_model_name()

st.title("Fake Job Detection")
st.write("This system predicts whether a job posting is real or fake using NLP and Machine Learning.")
st.info(f"Best trained model: {best_model_name}")

job_text = st.text_area("Enter job posting text", height=220)

if st.button("Predict"):
    if not job_text.strip():
        st.warning("Please enter a job description first.")
    else:
        cleaned = clean_text(job_text)
        pred = model.predict([cleaned])[0]

        st.subheader("Result")
        if int(pred) == 1:
            st.error("🚨 Fake Job Posting")
        else:
            st.success("✅ Real Job Posting")

        with st.expander("Processed text"):
            st.write(cleaned)