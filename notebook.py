# Kaggle Notebook: Map Charting — Classification + Improvements
# Goal: Predict both Category and Category:Misconception
# Copy-paste into a Kaggle Notebook and run.

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix

import lightgbm as lgb

# -------------------------
# Config
# -------------------------
SEED = 2025
N_SPLITS = 5
WORD_NGRAMS = (1,2)
CHAR_NGRAMS = (3,5)
MAX_FEATURES_WORD = 50000
MAX_FEATURES_CHAR = 30000
SVD_COMP = 150
USE_SVD = True

# -------------------------
# Utilities
# -------------------------
def read_data():
    kaggle_input = '/kaggle/input'
    if os.path.exists(kaggle_input):
        comp_dirs = list(Path(kaggle_input).glob('*map*'))
        if comp_dirs:
            base = comp_dirs[0]
            train = pd.read_csv(base / 'train.csv')
            test = pd.read_csv(base / 'test.csv')
            sample = pd.read_csv(base / 'sample_submission.csv')
        else:
            train = pd.read_csv('/kaggle/working/train.csv')
            test = pd.read_csv('/kaggle/working/test.csv')
            sample = pd.read_csv('/kaggle/working/sample_submission.csv')
    else:
        train = pd.read_csv('/mnt/data/train.csv')
        test = pd.read_csv('/mnt/data/test.csv')
        sample = pd.read_csv('/mnt/data/sample_submission.csv')
    return train, test, sample

def clean_text(s):
    if pd.isna(s):
        return ""
    t = str(s)
    t = t.replace("\\frac", " ").replace("\\", " ")
    t = t.replace("\n", " ").replace("\r", " ").strip()
    return t

# -------------------------
# Load
# -------------------------
train, test, sample = read_data()
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())

# -------------------------
# Set target
# -------------------------
target_col = "Category"
if target_col not in train.columns:
    raise ValueError(f"Expected target column '{target_col}' not found in train.csv")

# -------------------------
# Prepare text features
# -------------------------
for df in [train, test]:
    df["QuestionText"] = df["QuestionText"].fillna("")
    df["MC_Answer"] = df["MC_Answer"].fillna("")
    df["StudentExplanation"] = df["StudentExplanation"].fillna("")

for col in ["QuestionText", "MC_Answer", "StudentExplanation"]:
    train[f"{col}_clean"] = train[col].apply(clean_text)
    test[f"{col}_clean"] = test[col].apply(clean_text)

COMB_COL = "combined_text"
train[COMB_COL] = (
    train["QuestionText_clean"] + " " +
    train["MC_Answer_clean"] + " " +
    train["StudentExplanation_clean"]
).fillna("")
test[COMB_COL] = (
    test["QuestionText_clean"] + " " +
    test["MC_Answer_clean"] + " " +
    test["StudentExplanation_clean"]
).fillna("")

# Add simple numeric features
for df in [train, test]:
    df["exp_len"] = df["StudentExplanation_clean"].str.len()
    df["exp_word_count"] = df["StudentExplanation_clean"].str.split().apply(len)

# -------------------------
# Encode target
# -------------------------
le = LabelEncoder()
y = le.fit_transform(train[target_col].astype(str))
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

# -------------------------
# TF-IDF features
# -------------------------
print("Fitting TF-IDF...")
word_vectorizer = TfidfVectorizer(
    ngram_range=WORD_NGRAMS,
    max_features=MAX_FEATURES_WORD,
    analyzer="word",
    stop_words="english"
)
char_vectorizer = TfidfVectorizer(
    ngram_range=CHAR_NGRAMS,
    max_features=MAX_FEATURES_CHAR,
    analyzer="char"
)

Xw = word_vectorizer.fit_transform(train[COMB_COL])
Xw_test = word_vectorizer.transform(test[COMB_COL])
Xc = char_vectorizer.fit_transform(train[COMB_COL])
Xc_test = char_vectorizer.transform(test[COMB_COL])

num_feats = train[["exp_len","exp_word_count"]].values
num_feats_test = test[["exp_len","exp_word_count"]].values

X = hstack([Xw, Xc, csr_matrix(num_feats)])
X_test = hstack([Xw_test, Xc_test, csr_matrix(num_feats_test)])
print("TF-IDF stacked shape:", X.shape)

# Optional SVD
if USE_SVD:
    print("Applying TruncatedSVD...")
    SVD_COMP_USE = min(SVD_COMP, X.shape[1]-1)
    svd = TruncatedSVD(n_components=SVD_COMP_USE, random_state=SEED)
    X_svd = svd.fit_transform(X)
    X_test_svd = svd.transform(X_test)
    X_final = X_svd
    X_test_final = X_test_svd
    print("SVD shapes:", X_final.shape, X_test_final.shape)
else:
    X_final = X
    X_test_final = X_test

# -------------------------
# Model training
# -------------------------
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_probs = np.zeros((train.shape[0], num_classes))
test_probs = np.zeros((test.shape[0], num_classes, N_SPLITS))

for fold, (tr_idx, val_idx) in enumerate(folds.split(X_final, y)):
    print("Fold", fold+1)
    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": SEED + fold,
        "feature_pre_filter": False,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)

    bst = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )

    val_prob = bst.predict(X_val, num_iteration=bst.best_iteration)
    test_prob = bst.predict(X_test_final, num_iteration=bst.best_iteration)

    oof_probs[val_idx] = val_prob
    test_probs[:, :, fold] = test_prob

    val_preds = np.argmax(val_prob, axis=1)
    acc = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds, average="macro")
    print(f" Fold {fold+1} — Acc: {acc:.4f}, Macro F1: {f1:.4f}")

test_prob_mean = test_probs.mean(axis=2)

oof_preds = np.argmax(oof_probs, axis=1)
cv_acc = accuracy_score(y, oof_preds)
cv_f1 = f1_score(y, oof_preds, average="macro")
print("CV — Acc: %.4f, Macro F1: %.4f" % (cv_acc, cv_f1))

# -------------------------
# Submission
# -------------------------
sub = sample.copy()
pred_labels = le.inverse_transform(np.argmax(test_prob_mean, axis=1))

# Fill Category column
sub["Category"] = pred_labels

# Fill Category:Misconception column with placeholder
# TODO: Replace with proper Misconception model later
sub["Category:Misconception"] = "True_Correct:NA False_Neither:NA False_Misconception:Incomplete"

# Save
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Saved submission to", out_path)
print(sub.head())
