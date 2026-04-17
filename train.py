import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from preprocess import combine_text_columns

DATA_PATH = "/home/abdo/projects/fake-job-detection/data/fake_job_postings_clean.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    target_col = "fraudulent"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df[target_col].value_counts())

    X = combine_text_columns(df)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "svm": LinearSVC(
            class_weight="balanced",
            dual="auto"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
        "naive_bayes": MultinomialNB()
    }

    best_model_name = None
    best_pipeline = None
    best_fake_recall = -1.0
    best_f1_fake = -1.0

    for name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {name}")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ("clf", model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision_fake = precision_score(y_test, preds, pos_label=1, zero_division=0)
        recall_fake = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1_fake = f1_score(y_test, preds, pos_label=1, zero_division=0)

        print(f"Accuracy:        {acc:.4f}")
        print(f"Fake Precision:  {precision_fake:.4f}")
        print(f"Fake Recall:     {recall_fake:.4f}")
        print(f"Fake F1-score:   {f1_fake:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))

        if (recall_fake > best_fake_recall) or (
            recall_fake == best_fake_recall and f1_fake > best_f1_fake
        ):
            best_fake_recall = recall_fake
            best_f1_fake = f1_fake
            best_model_name = name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))

    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_model_name)

    print("\n" + "=" * 70)
    print("Best model saved successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Best fake-class Recall: {best_fake_recall:.4f}")
    print(f"Best fake-class F1-score: {best_f1_fake:.4f}")

if __name__ == "__main__":
    main()