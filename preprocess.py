import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)

def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    text_cols = [
        "title",
        "location",
        "department",
        "salary_range",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
    ]

    existing_cols = [col for col in text_cols if col in df.columns]

    if not existing_cols:
        raise ValueError("No expected text columns found in dataset.")

    combined = df[existing_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return combined.apply(clean_text)