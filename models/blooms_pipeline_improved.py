"""
=============================================================
  Bloom's Taxonomy Question Classifier - IMPROVED VERSION
  Traditional ML Pipeline: TF-IDF + Logistic Regression
  Categories: K1, K2, K3, K4, K5, K6, not_blooms
  
  IMPROVEMENTS:
  - Better keyword detection for each level
  - Enhanced feature engineering
  - Optimized hyperparameters
  - More balanced training data
=============================================================
"""

import pandas as pd, numpy as np, re, warnings, joblib, os # type: ignore
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion # type: ignore
from sklearn.preprocessing import FunctionTransformer # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay # type: ignore
import matplotlib; matplotlib.use('Agg') # type: ignore
import matplotlib.pyplot as plt # type: ignore

# ─────────────────────────────────────────────────────────────
# BLOOM'S TAXONOMY KEYWORDS - Enhanced for better distinction
# ─────────────────────────────────────────────────────────────
BLOOM_KEYWORDS = {
    'K1': ['define', 'list', 'name', 'state', 'identify', 'label', 'recall', 'recite', 
           'what is', 'who is', 'when did', 'where is', 'how many', 'how much', 
           'draw and label', 'make a list'],
    
    'K2': ['explain', 'describe', 'summarize', 'paraphrase', 'interpret', 'discuss',
           'in your own words', 'what happens when', 'what is meant by', 'the function of',
           'classify the following'],
    
    'K3': ['calculate', 'solve', 'apply', 'demonstrate', 'use', 'implement', 'construct',
           'find the', 'compute', 'show how', 'illustrate how', 'employ', 'manipulate'],
    
    'K4': ['analyze', 'compare', 'contrast', 'examine', 'differentiate', 'distinguish',
           'compare and contrast', 'similarities and differences', 'break down',
           'relationship between', 'how does', 'what factors'],
    
    'K5': ['evaluate', 'assess', 'judge', 'critique', 'justify', 'defend', 'argue',
           'which is better', 'which is more effective', 'what is the best',
           'strengths and weaknesses', 'recommend', 'is it ethical'],
    
    'K6': ['create', 'design', 'develop', 'compose', 'formulate', 'devise', 'invent',
           'propose', 'construct a new', 'develop a plan', 'write a story',
           'design an experiment', 'build a model', 'create a program']
}

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────
def extract_keyword_features(texts):
    """Extract keyword-based features for each Bloom level"""
    features = []
    for text in texts:
        text_lower = str(text).lower()
        feat_dict = {}
        
        for level, keywords in BLOOM_KEYWORDS.items():
            # Count how many keywords from this level appear
            count = sum(1 for kw in keywords if kw in text_lower)
            feat_dict[f'{level}_keywords'] = count
            
            # Binary: does ANY keyword from this level appear?
            feat_dict[f'{level}_has_keyword'] = 1 if count > 0 else 0
        
        # Additional features
        feat_dict['has_question_mark'] = 1 if '?' in text else 0
        feat_dict['starts_with_wh'] = 1 if any(text_lower.startswith(w) for w in ['what', 'where', 'when', 'who', 'why', 'how']) else 0
        feat_dict['word_count'] = len(text_lower.split())
        
        # Detect NOT_BLOOMS patterns
        not_blooms_patterns = [
            'how are you', 'how are you doing', 'what time', 'good morning', 
            'good afternoon', 'good evening', 'good night', 'hello', 'hi there',
            'bring me', 'pass me', 'give me', 'close the', 'open the',
            'turn on', 'turn off', 'wait here', 'sit down', 'stand up',
            'i am feeling', 'i feel', 'that was', 'that is',
            'nearest bus', 'nearest atm', 'nearest bank', 'nearest hospital',
            'where can i find', 'how do i get', 'which way to', 'where should i'
        ]
        feat_dict['has_not_blooms_pattern'] = 1 if any(p in text_lower for p in not_blooms_patterns) else 0
        
        # Detect imperative/command structure (typical of NOT_BLOOMS)
        command_words = ['please', 'bring', 'close', 'open', 'turn', 'sit', 'stand', 'wait', 'call', 'send']
        feat_dict['is_command'] = 1 if any(text_lower.startswith(w) or text_lower.startswith('please ' + w) for w in command_words) else 0
        
        features.append(list(feat_dict.values()))
    
    return np.array(features)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data(filepath):
    print("\n📂 STEP 1: Loading Improved Dataset...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip()
    df.dropna(subset=['Questions', 'Category'], inplace=True)
    print(f"   Rows: {len(df)}\n   Distribution:\n{df['Category'].value_counts().to_string()}")
    return df

# ─────────────────────────────────────────────────────────────
# STEP 2: CLEAN & DEDUPLICATE
# ─────────────────────────────────────────────────────────────
def clean_and_label(df):
    print("\n🧹 STEP 2: Cleaning & Deduplicating...")
    df = df.copy()
    df['Questions'] = df['Questions'].apply(lambda x: re.sub(r'\s+\d+\?$', '?', str(x)).strip())
    df = df.drop_duplicates(subset=['Questions'])
    print(f"   After dedup: {len(df)} rows")
    return df

# ─────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_text(text):
    """Light preprocessing - preserve important question words"""
    text = str(text).lower()
    # Don't remove numbers completely, they might be important
    text = re.sub(r'[^\w\s?]', ' ', text)  # Keep question marks
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_df(df):
    print("\n⚙️  STEP 3: Preprocessing...")
    df = df.copy()
    df['processed'] = df['Questions'].apply(preprocess_text)
    return df

# ─────────────────────────────────────────────────────────────
# STEP 4: EDA / ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyse_data(df, save_dir="."):
    print("\n📊 STEP 4: Analysing Dataset...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['Category'].value_counts()
    colors = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#FF5722','#9E9E9E']
    axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
    axes[0].set_title("Class Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Bloom Level"); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(counts.values): axes[0].text(i, v+5, str(v), ha='center', fontsize=9)
    df['q_len'] = df['processed'].apply(lambda x: len(x.split()))
    avg_len = df.groupby('Category')['q_len'].mean().sort_values()
    axes[1].barh(avg_len.index, avg_len.values, color='#4C72B0')
    axes[1].set_title("Avg Question Length by Level", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Avg Word Count")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "eda_analysis.png"), dpi=150); plt.close()
    print("   EDA chart saved.")

# ─────────────────────────────────────────────────────────────
# STEP 5 & 6: ENHANCED MODEL WITH FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def build_and_evaluate(df, save_dir="."):
    print("\n🤖 STEP 5 & 6: Enhanced TF-IDF + Keyword Features + Logistic Regression...")
    X, y = df['processed'], df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Enhanced pipeline with keyword features
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=12000,
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
                # Don't use stop words - action verbs are important!
                stop_words=None
            )),
            ('keywords', FunctionTransformer(extract_keyword_features, validate=False))
        ])),
        ('clf', LogisticRegression(
            max_iter=3000,
            C=2.0,
            solver='lbfgs',
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n   ✅ Test Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"   5-Fold CV       : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("Confusion Matrix – Bloom's Classifier (IMPROVED)", fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150); plt.close()

    # Save model
    model_path = os.path.join(save_dir, "blooms_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"   Model saved: {model_path}")
    return pipeline, acc, cv_scores

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\L E N O V O\OneDrive\Desktop\blooms_final\data\blooms_improved_dataset.csv"
    SAVE_DIR = os.path.dirname(__file__)

    df_raw   = load_data(DATASET_PATH)
    df_clean = clean_and_label(df_raw)
    df_proc  = preprocess_df(df_clean)
    analyse_data(df_proc, save_dir=SAVE_DIR)
    pipeline, accuracy, cv_scores = build_and_evaluate(df_proc, save_dir=SAVE_DIR)

    print("\n" + "="*60)
    print("🎓 BLOOM'S TAXONOMY CLASSIFIER — TRAINING COMPLETE (IMPROVED)")
    print("="*60)
    print(f"  Model   : Logistic Regression + TF-IDF + Keyword Features")
    print(f"  Test Acc: {accuracy*100:.2f}%")
    print(f"  CV Acc  : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Saved   : {os.path.join(SAVE_DIR, 'blooms_model.pkl')}")
    print("="*60)
