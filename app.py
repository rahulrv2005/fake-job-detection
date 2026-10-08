"""
SecureHire ML - Flask Backend REST API
Serves /predict, /dashboard, /history, and /model-comparison endpoints
"""

import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
from train_models import clean_text
from predict import check_indicators

app = Flask(__name__)
DB_PATH = os.path.join("database", "securehire.db")

def get_db_connection():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            company_name TEXT NOT NULL,
            location TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            indicators TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json() or {}
    title = data.get('title', '')
    company = data.get('company_name', '')
    location = data.get('location', 'Remote')
    description = data.get('description', '')
    requirements = data.get('requirements', '')
    benefits = data.get('benefits', '')

    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    model_path = os.path.join("models", "best_model.pkl")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return jsonify({"error": "Models not found. Run train_models.py first"}), 500

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    combined = f"{title} {title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined)
    vec = vectorizer.transform([cleaned])

    pred = int(model.predict(vec)[0])
    prob = model.predict_proba(vec)[0]

    confidence = round(float(max(prob)) * 100, 2)
    verdict = "FAKE" if pred == 1 else "REAL"
    indicators = check_indicators(combined)
    model_name = "Random Forest"

    # Save to SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (job_title, company_name, location, prediction, confidence, model_used, indicators, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, company, location, verdict, confidence, model_name, json.dumps(indicators), datetime.utcnow().isoformat()))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return jsonify({
        "success": True,
        "recordId": record_id,
        "jobTitle": title,
        "companyName": company,
        "prediction": verdict,
        "isFake": pred == 1,
        "confidence": confidence,
        "modelUsed": model_name,
        "indicators": indicators
    })

@app.route('/model-comparison', methods=['GET'])
def model_comparison():
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found. Run train_models.py first"}), 404

@app.route('/dashboard', methods=['GET'])
def dashboard():
    conn = get_db_connection()
    total = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    fakes = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'FAKE'").fetchone()[0]
    reals = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'REAL'").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0.0
    conn.close()

    return jsonify({
        "totalJobsChecked": total,
        "fakeJobsCount": fakes,
        "realJobsCount": reals,
        "averageConfidence": round(avg_conf, 1),
        "bestAlgorithm": "Random Forest"
    })

@app.route('/history', methods=['GET'])
def history():
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 50').fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

if __name__ == '__main__':
    print("[*] SecureHire ML Flask API running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
