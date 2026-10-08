"""
SecureHire ML - Standalone Python Predictor
Loads saved best_model.pkl and vectorizer.pkl to evaluate a job posting.
"""

import os
import joblib
from train_models import clean_text

SUSPICIOUS_RULES = [
    ("Upfront Registration Fee", ["registration fee", "application fee", "processing fee", "security deposit"]),
    ("Check Cashing Scam", ["cashier check", "deposit check", "purchase equipment with check"]),
    ("Unrealistic Income", ["guaranteed income", "earn $1000 daily", "work 1 hour earn", "easy money"]),
    ("Urgent Hiring Pressure", ["urgent hiring", "urgently needed today", "start today", "limited vacancies"]),
    ("Informal Contact", ["telegram", "whatsapp", "gmail.com", "yahoo.com"])
]

def check_indicators(text):
    lower = text.lower()
    flags = []
    for title, keywords in SUSPICIOUS_RULES:
        for kw in keywords:
            if kw in lower:
                flags.append(title)
                break
    return flags

def predict_job(title, company, description, requirements="", benefits=""):
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    model_path = os.path.join("models", "best_model.pkl")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print("[-] Models not found. Please run 'python train_models.py' first.")
        return

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    combined = f"{title} {title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)
    vec = vectorizer.transform([cleaned])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = round(max(prob) * 100, 2)
    verdict = "FAKE JOB" if pred == 1 else "REAL JOB"
    indicators = check_indicators(combined)

    print("\n" + "="*45)
    print("         SECUREHIRE ML PREDICTION")
    print("="*45)
    print(f"Title:       {title}")
    print(f"Company:     {company}")
    print(f"Verdict:     {verdict}")
    print(f"Confidence:  {confidence}%")
    print(f"Fake Prob:   {prob[1]*100:.2f}% | Real Prob: {prob[0]*100:.2f}%")
    if indicators:
        print("\n[!] Suspicious Indicators Flagged:")
        for ind in set(indicators):
            print(f"  • {ind}")
    else:
        print("\n[✓] No suspicious scam patterns flagged.")
    print("="*45)

if __name__ == "__main__":
    import sys
    test_title = "Remote Data Entry Assistant"
    test_company = "FastCash LLC"
    test_desc = "Urgent hiring! Guaranteed income of $1000 daily. Deposit our cashier check to buy supplies. Contact on Telegram @FastCashHR."
    predict_job(test_title, test_company, test_desc)
