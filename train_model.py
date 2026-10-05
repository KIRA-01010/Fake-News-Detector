"""
train_model.py
---------------
Trains a fake-news detection model.

Pipeline:
  1. Load labeled news text data with pandas.
  2. Split into train/test sets.
  3. Convert text to TF-IDF numeric features (scikit-learn).
  4. Train a PassiveAggressiveClassifier (a classic, fast, strong baseline
     for text classification / fake news detection) and compare it against
     Logistic Regression.
  5. Evaluate with accuracy, a classification report, and a confusion matrix
     plotted with matplotlib.
  6. Save the trained model + vectorizer to disk with joblib so predict.py
     can reuse them without retraining.

Usage:
    python train_model.py
    python train_model.py --data data/news_dataset.csv --test-size 0.2
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = "model"
OUTPUT_DIR = "outputs"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    print(f"Loaded {len(df)} rows from {path}")
    print(df["label"].value_counts())
    return df


def build_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def train_and_evaluate(X_train, y_train, X_test, y_test):
    candidates = {
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": clf, "accuracy": acc, "preds": preds}
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")
    return best_name, results


def plot_confusion_matrix(y_test, preds, labels, out_path):
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Best Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {out_path}")


def plot_class_distribution(df, out_path):
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of Articles")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved class distribution plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a fake news detection model.")
    parser.add_argument("--data", default="data/news_dataset.csv", help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for testing")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(args.data)
    plot_class_distribution(df, os.path.join(OUTPUT_DIR, "class_distribution.png"))

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=42, stratify=df["label"]
    )

    vectorizer, X_train, X_test = build_features(X_train_text, X_test_text)
    best_name, results = train_and_evaluate(X_train, y_train, X_test, y_test)
    best_model = results[best_name]["model"]
    best_preds = results[best_name]["preds"]

    labels = sorted(df["label"].unique())
    plot_confusion_matrix(y_test, best_preds, labels, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # Save model + vectorizer + metadata together
    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "model_name": best_name,
            "labels": labels,
        },
        os.path.join(MODEL_DIR, "fake_news_model.joblib"),
    )
    print(f"\nSaved trained model to {MODEL_DIR}/fake_news_model.joblib")


if __name__ == "__main__":
    main()
