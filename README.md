# Bloom's Taxonomy Classifier

An AI-powered Streamlit web app that classifies questions into Bloom's Taxonomy cognitive levels (K1-K6) and detects non-Bloom prompts (`not_blooms`).

This project combines NLP feature extraction (TF-IDF + engineered keyword features) with a Logistic Regression classifier to provide fast and interpretable predictions for educational question analysis.

## Project Highlights

- Classifies single questions and batch inputs
- Supports 7 output classes: `K1`, `K2`, `K3`, `K4`, `K5`, `K6`, `not_blooms`
- Shows prediction confidence and full probability breakdown
- Includes classification history with CSV export
- Includes CSV batch upload and downloadable results
- Achieves strong evaluation metrics on a labeled dataset

## Demo Features

### Single Question Classification
- Enter any educational question
- Get predicted level + confidence score
- View class-wise probability bars

### Batch Classification
- Paste multiple questions (one per line), or
- Upload a CSV with a `Questions` column
- Download classified output as CSV

### History Tracking
- Stores current-session predictions
- Export history to CSV

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

## Model Information

| Property | Value |
|---|---|
| Model | Logistic Regression |
| Text Features | TF-IDF (`1-3` n-grams) |
| Engineered Features | Bloom keyword counts, WH-start flag, command/non-Bloom patterns, etc. |
| Test Accuracy | **93.18%** |
| 5-Fold CV Accuracy | **91.34% +/- 2.62%** |
| Dataset Size | 438 labeled questions |
| Classes | `K1`, `K2`, `K3`, `K4`, `K5`, `K6`, `not_blooms` |

## Bloom's Taxonomy Levels

| Level | Name | Typical Verbs |
|---|---|---|
| K1 | Remember | Define, List, Name, Identify |
| K2 | Understand | Explain, Describe, Summarize, Discuss |
| K3 | Apply | Calculate, Solve, Demonstrate, Use |
| K4 | Analyze | Analyze, Compare, Contrast, Examine |
| K5 | Evaluate | Evaluate, Assess, Judge, Critique |
| K6 | Create | Create, Design, Develop, Compose |

## Project Structure

```text
blooms_final/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- blooms_improved_dataset.csv
`-- models/
    `-- blooms_pipeline_improved.py
```

> Note: The app expects the trained model file at `models/blooms_model.pkl`.

## Installation and Setup

### 1) Clone the repository
```bash
git clone https://github.com/maneeswar06/Blooms-Taxonomy.git
cd Blooms-Taxonomy
```

### 2) (Optional but recommended) Create virtual environment
```bash
python -m venv .venv
```

Activate:

- Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

- macOS/Linux:
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the Streamlit app
```bash
streamlit run app.py
```

## Training / Retraining the Model

If you update `data/blooms_improved_dataset.csv`, retrain the model:

```bash
python models/blooms_pipeline_improved.py
```

This will generate:
- `models/blooms_model.pkl` (trained model)
- EDA and confusion-matrix images in the `models/` directory

## CSV Input Format for Batch Mode

Use a CSV file with a column named exactly:

```text
Questions
```

Example:

```csv
Questions
What is photosynthesis?
Explain the process of osmosis.
Design an experiment to test plant growth.
```

## Future Improvements

- Add deep learning transformer baseline for comparison
- Deploy publicly (Streamlit Community Cloud / Render)
- Add unit tests for preprocessing and prediction utilities
- Add confusion matrix and EDA images directly to this README

## Authors

Built with teamwork by:

- **Purnithaa B R**
- **Maneeswar KG**

If this project helped you, consider giving the repo a star.
