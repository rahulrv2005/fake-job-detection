"""
SecureHire ML - Standalone Python Model Trainer
Trains 5 Algorithms (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM)
Evaluates holdout performance (Accuracy, Precision, Recall, F1-score)
Selects the best performing model dynamically and saves .pkl files.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_evaluate():
    dataset_path = os.path.join("dataset", "fake_job_postings.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    print(f"[*] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Combine text features
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    print("[*] Preprocessing and combining text columns...")
    df['combined_text'] = (
        df['title'] + " " + df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    ).apply(clean_text)

    X = df['combined_text']
    y = df['fraudulent'].fillna(0).astype(int)

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[*] Train set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Define 5 ML Algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=12),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "SVM": SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    }

    os.makedirs("models", exist_ok=True)
    results = {}

    print("\n" + "="*65)
    print("                MODEL PERFORMANCE EVALUATION")
    print("="*65)
    print(f"{'Algorithm':<22} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
    print("-"*65)

    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds, zero_division=0) * 100
        rec = recall_score(y_test, preds, zero_division=0) * 100
        f1 = f1_score(y_test, preds, zero_division=0) * 100

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        results[name] = {
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        }

        # Save individual model
        file_key = name.lower().replace(" ", "_")
        joblib.dump(clf, os.path.join("models", f"{file_key}.pkl"))

        print(f"{name:<22} {acc:>7.2f}% {prec:>8.2f}% {rec:>8.2f}% {f1:>8.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf

    print("="*65)
    print(f"\n[+] BEST PERFORMING ALGORITHM: {best_model_name} (F1-Score: {best_f1:.2f}%)")

    # Save vectorizer and best model
    joblib.dump(vectorizer, os.path.join("models", "vectorizer.pkl"))
    joblib.dump(best_model_obj, os.path.join("models", "best_model.pkl"))

    # Save evaluation summary to JSON
    with open(os.path.join("models", "evaluation_metrics.json"), "w") as f:
        json.dump({
            "best_algorithm": best_model_name,
            "best_f1": best_f1,
            "metrics": results
        }, f, indent=2)

    print("[+] All models saved successfully inside 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()
