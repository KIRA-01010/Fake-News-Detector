"""
predict.py
----------
Loads the trained fake-news model + TF-IDF vectorizer (saved by
train_model.py) and classifies new text as REAL or FAKE.

Usage:
    python predict.py
        -> interactive mode, type text and press Enter (Ctrl+C to quit)

    python predict.py --text "Some headline or article text here"
        -> classify a single piece of text

    python predict.py --csv path/to/articles.csv
        -> batch mode; reads a 'text' column and writes
           path/to/articles_predictions.csv with 'prediction' and
           'confidence' columns
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("model", "fake_news_model.joblib")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. Run `python train_model.py` first."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["vectorizer"], bundle["model_name"], bundle["labels"]


def predict_texts(texts, model, vectorizer):
    """Return (predictions, confidence_scores) for a list of texts."""
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    # Not all classifiers expose predict_proba (PassiveAggressiveClassifier
    # doesn't), so fall back to a normalized decision_function margin as a
    # confidence proxy when needed.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        confidences = probs.max(axis=1)
    else:
        scores = model.decision_function(X)
        scores = np.atleast_1d(scores) if scores.ndim == 0 else scores
        # squash the raw margin into a 0.5-1.0 "confidence-like" score
        confidences = 1 / (1 + np.exp(-np.abs(scores)))

    return preds, confidences


def run_interactive(model, vectorizer):
    print("Fake News Detector - interactive mode (Ctrl+C to quit)")
    print("-" * 50)
    try:
        while True:
            text = input("\nEnter news text: ").strip()
            if not text:
                continue
            preds, confidences = predict_texts([text], model, vectorizer)
            print(f"  -> Prediction: {preds[0]}   (confidence: {confidences[0]:.2%})")
    except KeyboardInterrupt:
        print("\nGoodbye!")


def run_single(text, model, vectorizer):
    preds, confidences = predict_texts([text], model, vectorizer)
    print(f"Text: {text}")
    print(f"Prediction: {preds[0]}   (confidence: {confidences[0]:.2%})")


def run_batch(csv_path, model, vectorizer):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    texts = df["text"].astype(str).fillna("")
    preds, confidences = predict_texts(texts.tolist(), model, vectorizer)

    df["prediction"] = preds
    df["confidence"] = confidences

    base, ext = os.path.splitext(csv_path)
    out_path = f"{base}_predictions{ext}"
    df.to_csv(out_path, index=False)
    print(f"Wrote predictions for {len(df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify news text as REAL or FAKE.")
    parser.add_argument("--text", help="A single piece of text to classify.")
    parser.add_argument("--csv", help="Path to a CSV file with a 'text' column for batch prediction.")
    args = parser.parse_args()

    model, vectorizer, model_name, labels = load_model()
    print(f"Loaded model: {model_name} (labels: {labels})")

    if args.text:
        run_single(args.text, model, vectorizer)
    elif args.csv:
        run_batch(args.csv, model, vectorizer)
    else:
        run_interactive(model, vectorizer)


if __name__ == "__main__":
    main()
