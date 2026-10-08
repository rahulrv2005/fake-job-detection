# SecureHire ML: Combating Employment Scams Using Machine Learning

> **Fake Job Detection Using Machine Learning**  
> An end-to-end Machine Learning web application comparing 5 classical classification algorithms with TF-IDF feature extraction, holdout evaluation, SQLite persistence, and an interactive real-time dashboard.

---

## Table of Contents
1. [Project Overview & Workflow](#1-project-overview--workflow)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Step-by-Step Setup in VS Code](#3-step-by-step-setup-in-vs-code)
4. [Dataset Creation & Management](#4-dataset-creation--management)
5. [Model Training & 5 Algorithms](#5-model-training--5-algorithms)
6. [Running the Full-Stack Application](#6-running-the-full-stack-application)
7. [Step-by-Step GitHub Setup](#7-step-by-step-github-setup)
8. [API Endpoints Reference](#8-api-endpoints-reference)

---

## 1. Project Overview & Workflow

```
Job Details Input
       ↓
Data Preprocessing (Text Cleaning, Lowercasing, Stopword Removal, Stemming)
       ↓
TF-IDF Vectorization (Top Informative n-grams)
       ↓
Stratified 80/20 Train/Test Split
       ↓
┌───────────────────────┬───────────────────┬─────────────────────┐
↓                       ↓                   ↓                     ↓
Logistic Regression   Decision Tree     Random Forest        Naive Bayes / SVM
└───────────────────────┴───────────────────┴─────────────────────┘
       ↓
Holdout Evaluation (Accuracy, Precision, Recall, F1-Score)
       ↓
Dynamic Best Algorithm Selection (Highest F1-Score)
       ↓
Prediction (REAL / FAKE) + Calibrated Confidence Score (%)
       ↓
Suspicious Red-Flag Indicators Extraction
       ↓
SQLite Database (database/securehire.db) & Live Dashboard
```

---

## 2. Project Folder Structure

```
SecureHire-ML/
│
├── dataset/
│   ├── fake_job_postings.csv      # Kaggle EMSCAD structured dataset
│   ├── seed_data.ts               # Core authentic job templates
│   └── generator.ts               # Synthetic stratified data generator
│
├── ml/
│   ├── preprocessor.ts            # Text normalization & stopwords
│   ├── vectorizer.ts              # TF-IDF calculation & L2 normalization
│   ├── metrics.ts                 # Accuracy, Precision, Recall, F1, Confusion Matrix
│   ├── indicators.ts              # Red-flag rule extraction
│   ├── trainer.ts                 # Master 5-model pipeline & best selector
│   └── algorithms/
│       ├── logistic_regression.ts # Sigmoid & L2 gradient descent
│       ├── decision_tree.ts       # Gini impurity tree splits
│       ├── random_forest.ts       # Ensemble bagging & feature sampling
│       ├── naive_bayes.ts         # Multinomial NB with Laplace smoothing
│       └── svm.ts                 # Linear SVM with Platt scaling
│
├── database/
│   ├── db.ts                      # SQLite manager using sql.js
│   └── securehire.db              # Persistent SQLite database file
│
├── models/                        # Serialized weights, trees, and metrics
│   ├── logistic_regression.json / .pkl
│   ├── decision_tree.json / .pkl
│   ├── random_forest.json / .pkl
│   ├── naive_bayes.json / .pkl
│   ├── svm.json / .pkl
│   ├── best_model.json / .pkl
│   └── evaluation_metrics.json
│
├── src/                           # React frontend (Vite + Tailwind CSS)
│   ├── components/
│   │   ├── Navbar.tsx             # Typographic navigation
│   │   ├── Hero.tsx               # Architecture diagram & quick stats
│   │   ├── JobDetectionForm.tsx   # Input form with 1-click samples
│   │   ├── PredictionResult.tsx   # Verdict shield & 5-model consensus
│   │   ├── ModelComparison.tsx    # Benchmark table & confusion matrices
│   │   ├── DashboardStats.tsx     # Live SQLite telemetry charts
│   │   └── HistoryView.tsx        # Records table with search, filter, CSV
│   ├── App.tsx                    # Main application router
│   ├── types.ts                   # TypeScript interfaces
│   └── main.tsx                   # React mount
│
├── server.ts                      # Express server with Vite middleware (Port 3000)
├── app.py                         # Standalone Python Flask backend
├── train_models.py                # Standalone Scikit-Learn training script
├── predict.py                     # Standalone Python prediction test
├── requirements.txt               # Python dependencies
├── package.json                   # Node.js dependencies
└── README.md                      # Project documentation
```

---

## 3. Step-by-Step Setup in VS Code

### Prerequisites:
1. **VS Code**: Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. **Node.js** (v18 or higher): Download from [nodejs.org](https://nodejs.org/)
3. **Python 3.10+** (Optional, if running Python backend): Download from [python.org](https://www.python.org/)
4. **Git**: Download from [git-scm.com](https://git-scm.com/)

---

### Step 3.1 — Open Project in VS Code
1. Open VS Code.
2. Click **File > Open Folder...**
3. Select your `SecureHire-ML` project folder.
4. Open the integrated terminal in VS Code:
   - Shortcut: ``Ctrl + ` `` (Windows/Linux) or ``Cmd + ` `` (macOS).

---

### Step 3.2 — Install Node.js Dependencies
In the VS Code terminal, run:
```bash
npm install
```
This installs:
- Express & Vite
- React 19 & Lucide Icons
- Tailwind CSS
- sql.js (WebAssembly SQLite)
- canvas-confetti

---

### Step 3.3 — (Optional) Set Up Python Environment
If you wish to run the Scikit-learn training script (`train_models.py`):
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Creation & Management

### The EMSCAD Dataset Schema:
The dataset file is located at `dataset/fake_job_postings.csv` with standard Kaggle EMSCAD fields:
- `title`: Job Title
- `company_profile`: Corporate background description
- `description`: Core duties and work scope
- `requirements`: Required experience and qualifications
- `benefits`: Insurance, salary structure, perks
- `salary_range`: Compensation bracket
- `employment_type`: Full-time, Part-time, Contract
- `fraudulent`: **`0` = Real Job, `1` = Fake/Scam Job**

### Generating or Refreshing the Dataset:
To regenerate `dataset/fake_job_postings.csv` with balanced real and fraudulent postings:
```bash
npx tsx -e "
import fs from 'fs';
import { getFullDataset } from './dataset/generator.ts';
const data = getFullDataset();
const headers = ['job_id','title','company_profile','description','requirements','benefits','salary_range','employment_type','required_experience','required_education','industry','function','fraudulent'];
const escapeCsv = (str) => '\"' + String(str || '').replace(/\"/g, '\"\"').replace(/\n/g, ' ') + '\"';
const rows = [headers.join(',')];
data.forEach((item, index) => {
  rows.push([index + 1, escapeCsv(item.title), escapeCsv(item.company_profile), escapeCsv(item.description), escapeCsv(item.requirements), escapeCsv(item.benefits), escapeCsv(item.salary_range), escapeCsv(item.employment_type), escapeCsv(item.required_experience), escapeCsv(item.required_education), escapeCsv(item.industry), escapeCsv(item.function), item.fraudulent ?? 0].join(','));
});
fs.writeFileSync('dataset/fake_job_postings.csv', rows.join('\n'));
console.log('Dataset generated successfully: ' + data.length + ' records.');
"
```

---

## 5. Model Training & 5 Algorithms

### Option A: Automatic Training (Full-Stack Engine)
When the full-stack server starts, it automatically:
1. Loads the dataset.
2. Cleanses and stems all text.
3. Calculates TF-IDF matrices (up to 800 n-grams).
4. Splits 80% train / 20% test using stratified sampling.
5. Trains all 5 algorithms and evaluates holdout metrics.
6. Automatically flags the best algorithm based on holdout F1-score.
7. Saves models to `models/`.

### Option B: Python Scikit-Learn Training
Run the standalone training script in your activated Python environment:
```bash
python train_models.py
```
**Sample Output:**
```
=================================================================
                MODEL PERFORMANCE EVALUATION
=================================================================
Algorithm              Accuracy   Precision   Recall     F1-Score  
-----------------------------------------------------------------
Logistic Regression     100.00%    100.00%    100.00%    100.00%
Decision Tree            96.00%    100.00%     91.67%     95.65%
Random Forest           100.00%    100.00%    100.00%    100.00%
Naive Bayes             100.00%    100.00%    100.00%    100.00%
SVM                     100.00%    100.00%    100.00%    100.00%
=================================================================
[+] BEST PERFORMING ALGORITHM: Random Forest (F1-Score: 100.00%)
[+] All models saved successfully inside 'models/' directory.
```

### Test Python Inference from Terminal:
```bash
python predict.py
```

---

## 6. Running the Full-Stack Application

### Start the Development Server:
```bash
npm run dev
```

1. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```
2. **Features available immediately in the UI:**
   - **Home**: View architecture flow and overview.
   - **Detect Job**: Fill the form or click any demo sample button (**Stripe Real**, **Nurse Real**, **Cashier Check Scam**, **Wire Mule Scam**, **Registration Fee Scam**).
   - **5 ML Models**: Inspect the real-time holdout comparison table and confusion matrices; click **Retrain All Models** to run fresh training.
   - **Dashboard**: Live SQLite statistics and scam pattern frequency chart.
   - **History**: Query the SQLite database, filter records, delete entries, and export history as CSV.

---

## 7. Step-by-Step GitHub Setup

Follow these exact steps to push your project to GitHub:

### Step 7.1 — Initialize Git Locally
Open your terminal inside the project root:
```bash
# Initialize git repository
git init

# Check status
git status
```

### Step 7.2 — Stage and Commit Files
```bash
# Add all files to staging
git add .

# Create initial commit
git commit -m "feat: initial commit for SecureHire ML Fake Job Detection"
```

### Step 7.3 — Create a New Repository on GitHub
1. Go to [github.com](https://github.com/) and log in.
2. In the top right corner, click **+ > New repository**.
3. Set **Repository name**: `SecureHire-ML`
4. Set description: `Combating Employment Scams Using Machine Learning (5 Algorithms + TF-IDF)`
5. Choose **Public** or **Private**.
6. **DO NOT** check "Add a README file" (we already created one).
7. Click **Create repository**.

### Step 7.4 — Link and Push to GitHub
Copy the commands shown on your GitHub repository page and run them in your VS Code terminal:
```bash
# Rename default branch to main
git branch -M main

# Add your GitHub remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/SecureHire-ML.git

# Push the code to GitHub
git push -u origin main
```

---

## 8. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Analyzes a job posting, computes confidence, and stores verdict in SQLite |
| `GET` | `/api/model-comparison` | Returns holdout metrics (Accuracy, Precision, Recall, F1) for all 5 models |
| `POST` | `/api/retrain` | Retrains all 5 models on the dataset and returns fresh metrics |
| `GET` | `/api/dashboard` | Aggregates live statistics (Real %, Fake %, Avg Confidence) from SQLite |
| `GET` | `/api/history` | Fetches historical prediction records from `database/securehire.db` |
| `DELETE`| `/api/history/:id` | Deletes a specific prediction record from SQLite |
| `DELETE`| `/api/history` | Clears all prediction history from SQLite |
| `GET` | `/api/samples` | Provides preloaded test job postings for 1-click verification |

---

## 9. Verification & Testing

To test that everything compiles and runs with zero issues:
```bash
# Run TypeScript compilation and build check
npm run build

# Run TypeScript linter
npm run lint
```
Both commands will return with `0` exit code.
