# Fake News Detection System

A complete, runnable Python project that trains a machine learning model to
classify news text as **REAL** or **FAKE**, built with:

- **pandas** – loading and cleaning the dataset
- **NumPy** – numeric operations on prediction scores
- **Matplotlib** – dataset and evaluation visualizations
- **scikit-learn** – TF-IDF text vectorization + classification models

## Project structure

```
fake_news_detector/
├── data/
│   ├── generate_dataset.py   # builds the demo dataset
│   └── news_dataset.csv      # generated demo dataset (text, label)
├── model/
│   └── fake_news_model.joblib  # saved trained model (created by train_model.py)
├── outputs/
│   ├── class_distribution.png  # bar chart of REAL vs FAKE counts
│   └── confusion_matrix.png    # confusion matrix of the best model
├── train_model.py            # trains and evaluates the model
├── predict.py                # loads the model and classifies new text
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. About the dataset

`data/news_dataset.csv` is a **synthetic demo dataset** (800 rows, generated
by `data/generate_dataset.py`) that mimics two common styles:

- **REAL** — neutral, attributed, fact-style sentences ("According to
  Reuters, officials confirmed...")
- **FAKE** — sensational, clickbait-style sentences ("SHOCKING: you won't
  believe...")

This lets the whole pipeline run immediately with no downloads. Because the
two styles are quite distinct, the demo model reaches very high accuracy —
that's expected for synthetic data, not a claim about real-world performance.

**For a real project, swap in a genuine dataset**, such as the Kaggle
["Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
or the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip).
Just make sure your CSV has two columns: `text` and `label` (with label
values `REAL` / `FAKE`), and pass it in with `--data`:

```bash
python train_model.py --data path/to/your_real_dataset.csv
```

To regenerate the demo dataset yourself:

```bash
python data/generate_dataset.py
```

## 3. Train the model

```bash
python train_model.py
```

This will:
1. Load and clean the dataset with pandas.
2. Split it into train/test sets (80/20 by default).
3. Convert article text into TF-IDF vectors (word + bigram features).
4. Train and compare two classifiers: `PassiveAggressiveClassifier` (a
   classic fast baseline for this task) and `LogisticRegression`.
5. Print accuracy and a full classification report for each.
6. Save two charts to `outputs/`: class distribution and confusion matrix.
7. Save the best-performing model + vectorizer to `model/fake_news_model.joblib`.

## 4. Classify news text

**Interactive mode** — type headlines/articles one at a time:

```bash
python predict.py
```

**Single prediction:**

```bash
python predict.py --text "Scientists confirm new findings after a two-year study published in a peer-reviewed journal."
```

**Batch prediction** from a CSV with a `text` column:

```bash
python predict.py --csv path/to/articles.csv
# writes path/to/articles_predictions.csv with 'prediction' and 'confidence' columns
```

## 5. How it works (short version)

1. **TF-IDF vectorization** turns each article's words into numeric features
   that reflect how distinctive each word/phrase is for that article,
   relative to the whole dataset.
2. A **linear classifier** learns a decision boundary between REAL and FAKE
   based on those features.
3. At prediction time, new text is transformed with the *same* fitted
   vectorizer and passed through the trained model to get a label and a
   confidence score.

## 6. Extending this project

- Swap in a real, larger dataset for meaningful real-world accuracy.
- Try additional models: `MultinomialNB`, `RandomForestClassifier`, or an
  SVM (`sklearn.svm.LinearSVC`).
- Add more features beyond TF-IDF: article length, punctuation/caps ratio,
  source credibility, etc.
- Wrap `predict.py`'s logic in a small Flask/FastAPI app for a web interface.
