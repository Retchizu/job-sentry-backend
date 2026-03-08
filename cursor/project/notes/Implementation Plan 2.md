# Scam Job Post Detection - Implementation Plan 2 (Second Dataset)

## Overview

This plan covers preprocessing, feature engineering, training, and evaluation for the **second dataset** (`processed_labeled_dataset_without_encoding.xlsx`). The first dataset (`fake_job_postings.csv`) has already been preprocessed and trained. This plan follows the same phased structure as Implementation Plan 1 but is adapted to the second dataset's column schema, data characteristics, and preprocessing needs.

### Dataset 2 Summary

- **File**: `processed_labeled_dataset_without_encoding.xlsx`
- **Rows**: 54,391
- **Columns**: 10

| # | Column          | Non-Null Count | Dtype   |
|---|-----------------|----------------|---------|
| 0 | job_title       | 54,389         | object  |
| 1 | location        | 54,391         | int64   |
| 2 | industry        | 54,391         | int64   |
| 3 | salary_range    | 17,123         | object  |
| 4 | company_profile | 45,992         | object  |
| 5 | job_desc        | 50,871         | object  |
| 6 | skills_desc     | 41,522         | object  |
| 7 | employment_type | 38,823         | object  |
| 8 | fraudulent      | 44,045         | float64 |
| 9 | text            | 54,391         | object  |

### Column Mapping (Dataset 1 → Dataset 2)

| Dataset 1 Column     | Dataset 2 Column | Notes                                            |
|----------------------|------------------|--------------------------------------------------|
| title                | job_title        | Rename needed for consistency                    |
| location             | location         | **Already label-encoded (int64) in Dataset 2**   |
| industry             | industry         | **Already label-encoded (int64) in Dataset 2**   |
| salary_range         | salary_range     | Same name; both sparse (~16% vs ~31% filled)     |
| company_profile      | company_profile  | Same name; Dataset 2 has ~85% fill rate           |
| description          | job_desc         | Rename needed; core text field                    |
| requirements         | skills_desc      | Rename needed; Dataset 2 calls it skills_desc     |
| employment_type      | employment_type  | Same name                                         |
| fraudulent           | fraudulent       | **Dataset 2 has ~10,346 nulls + float dtype**     |
| —                    | text             | **New column**: pre-concatenated text field        |
| department           | —                | Not available in Dataset 2                        |
| benefits             | —                | Not available in Dataset 2                        |
| telecommuting        | —                | Not available in Dataset 2                        |
| has_company_logo     | —                | Not available in Dataset 2                        |
| has_questions        | —                | Not available in Dataset 2                        |
| required_experience  | —                | Not available in Dataset 2                        |
| required_education   | —                | Not available in Dataset 2                        |
| function             | —                | Not available in Dataset 2                        |
| job_id               | —                | Not available in Dataset 2                        |

### Key Challenges Specific to Dataset 2

1. **~10,346 rows have no `fraudulent` label** — must be dropped or handled (semi-supervised learning)
2. **`location` and `industry` are pre-encoded as integers** — original text lost; cannot extract text-based features from them
3. **Fewer structural/metadata columns** — no `has_company_logo`, `has_questions`, `telecommuting`, `benefits`, `department`, `required_experience`, `required_education`
4. **`text` column exists** — appears to be a pre-concatenated field; need to verify its composition
5. **Larger dataset (54k vs 18k)** — more data but also more nulls to handle

---

## Phase 1: Data Loading and Exploration

### 1.1 Load Dataset

- Load `processed_labeled_dataset_without_encoding.xlsx` using `pandas.read_excel()`
- Verify shape, dtypes, and null counts match the expected schema above
- Display first few rows and `.info()` / `.describe()` outputs

### 1.2 Exploratory Data Analysis (EDA)

- **Label distribution**: Count of fraudulent (1) vs legitimate (0) vs null
  - Determine class imbalance ratio
  - Decide strategy for the ~10,346 unlabeled rows (drop vs semi-supervised)
- **Null analysis per column**:
  - `salary_range`: 68.5% missing — consider binary "has_salary" feature instead of parsing
  - `skills_desc`: 23.6% missing — impute with empty string or "Not specified"
  - `company_profile`: 15.4% missing — impute or flag
  - `job_desc`: 6.5% missing — impute with empty string
  - `employment_type`: 28.6% missing — impute with "Unknown"
  - `job_title`: 2 missing — drop those rows
- **Text column investigation**:
  - Check if `text` is a concatenation of `job_title + job_desc + skills_desc + company_profile`
  - Compare a few rows manually to understand composition
  - Decide whether to use `text` directly or reconstruct from individual columns
- **Pre-encoded columns**:
  - Check unique value counts for `location` (int64) and `industry` (int64)
  - Understand the encoding range and distribution
  - Note: original text labels are lost; these can only be used as categorical features, not for text analysis

### 1.3 Data Cleaning

- **Drop unlabeled rows**: Remove rows where `fraudulent` is NaN (reduces to ~44,045 rows)
  - Alternative: keep unlabeled rows for semi-supervised approaches (document decision)
- **Convert `fraudulent` to int**: Cast from float64 to int (0 or 1) after dropping NaN
- **Drop rows with missing `job_title`** (only 2 rows)
- **Fill missing text fields**:
  - `job_desc` → empty string `""`
  - `skills_desc` → empty string `""`
  - `company_profile` → empty string `""`
  - `employment_type` → `"Unknown"`
  - `salary_range` → `"Not specified"` or keep as NaN for binary feature
- **Remove duplicates**: Check for exact duplicate rows and drop them

**Deliverables**:

- Cleaned DataFrame with no NaN in critical columns
- EDA visualizations (label distribution, null heatmap, text length distributions)
- Documented decisions on null handling and unlabeled rows

**Tools**: Google Colab, pandas, openpyxl (for .xlsx reading), matplotlib/seaborn

---

## Phase 2: Data Preprocessing

### 2.1 Text Cleaning

Apply to `job_title`, `job_desc`, `skills_desc`, `company_profile`, and `text`:

- Remove HTML tags (common in job descriptions)
- Remove URLs and email addresses
- Remove special characters and excessive punctuation
- Normalize whitespace (collapse multiple spaces/newlines)
- Handle encoding issues (UTF-8 normalization)
- Remove non-printable characters

### 2.2 Text Normalization

- Lowercasing all text fields
- Expand contractions ("don't" → "do not", "we're" → "we are")
- Stemming/lemmatization (use spaCy or NLTK lemmatizer)
- Remove English stop words (with domain-aware exceptions — keep words like "free", "guaranteed" that may be scam indicators)

### 2.3 Reconstruct Combined Text Field

- Decide on the final combined text column:
  - Option A: Use existing `text` column as-is (if verified to be a good concatenation)
  - Option B: Create new `combined_text` = `job_title + " " + job_desc + " " + skills_desc + " " + company_profile`
- Apply all cleaning and normalization to the chosen combined field
- This will be the primary input for text-based models

### 2.4 Handle Categorical Columns

- **`employment_type`**: One-hot encode or label encode (values like Full-time, Part-time, Contract, etc.)
- **`location`** (int64): Already encoded — keep as-is for ML models
- **`industry`** (int64): Already encoded — keep as-is for ML models
- **`salary_range`**: Parse into min/max salary if format allows, otherwise create binary `has_salary` feature

### 2.5 Train/Validation/Test Split

- Split ratio: 70% train / 15% validation / 15% test
- Use **stratified** split to maintain class distribution
- Save split indices for reproducibility
- Ensure no data leakage between splits

**Deliverables**:

- Preprocessing functions (reusable for inference pipeline)
- Preprocessed DataFrame with cleaned text columns
- Train/validation/test splits stored in separate DataFrames

**Tools**: Google Colab, pandas, NLTK, spaCy, scikit-learn (train_test_split)

---

## Phase 3: Feature Engineering

### 3.1 Text-Based Features

Since Dataset 2 has fewer structured metadata columns, **text features become even more critical**:

- **Text length features**:
  - Word count, character count, sentence count for `job_desc`, `skills_desc`, `company_profile`
  - Average word length
  - Ratio of uppercase characters
- **Scam keyword indicators**:
  - Count of scam-associated keywords ("easy money", "work from home", "no experience needed", "guaranteed", "earn thousands", "urgent hiring", etc.)
  - Presence of excessive urgency language
- **Grammar and quality signals**:
  - Exclamation mark count
  - ALL CAPS word count
  - Spelling error density (optional, computationally expensive)
- **Sentiment features**:
  - Sentiment polarity score (TextBlob or VADER)
  - Subjectivity score

### 3.2 Structural Features (Adapted for Dataset 2)

Available structural features (fewer than Dataset 1):

| Feature                        | Source Column      | Description                                  |
|-------------------------------|--------------------|----------------------------------------------|
| `has_salary`                  | salary_range       | Binary: whether salary info is provided      |
| `has_company_profile`         | company_profile    | Binary: whether company profile exists       |
| `has_skills_desc`             | skills_desc        | Binary: whether skills/requirements provided |
| `job_desc_length`             | job_desc           | Length of job description                    |
| `skills_desc_length`          | skills_desc        | Length of skills description                 |
| `company_profile_length`      | company_profile    | Length of company profile                    |
| `employment_type_encoded`     | employment_type    | Encoded employment type                      |
| `location_encoded`            | location           | Pre-encoded location (int64)                 |
| `industry_encoded`            | industry           | Pre-encoded industry (int64)                 |

**Not available** (present in Dataset 1 only):
- `telecommuting`, `has_company_logo`, `has_questions`, `required_experience`, `required_education`, `department`, `function`

### 3.3 Text Vectorization

- **TF-IDF**:
  - Apply to combined text field
  - Tune `max_features`, `ngram_range`, `min_df`, `max_df`
  - Use same parameters as Dataset 1 for comparability where possible
- **Word Embeddings**:
  - Word2Vec / GloVe embeddings (pre-trained or trained on corpus)
  - Average word embeddings per document
- **Tokenization for Deep Learning**:
  - Tokenize for Bi-LSTM (Keras Tokenizer + padding)
  - Tokenize for DistilBERT (Hugging Face tokenizer)

### 3.4 Feature Matrix Construction

- Combine structural features + TF-IDF vectors for traditional ML
- Keep tokenized sequences separate for deep learning models
- Save feature matrices to avoid recomputation

**Deliverables**:

- Feature extraction functions
- Feature importance analysis
- Combined feature matrices for ML and DL pipelines

**Tools**: Google Colab, scikit-learn, NLTK, spaCy, gensim (Word2Vec), Hugging Face tokenizers

---

## Phase 4: Model Training

### 4.1 Traditional Machine Learning Models

Train the same model families as Dataset 1 for consistency:

- **Logistic Regression** (baseline)
- **Naive Bayes** (MultinomialNB for TF-IDF)
- **Support Vector Machine** (SVM with linear/RBF kernel)
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting (XGBoost)**

Training details:
- Input: TF-IDF features + structural features
- Hyperparameter tuning via GridSearchCV or RandomizedSearchCV
- 5-fold stratified cross-validation
- Handle class imbalance: SMOTE, class weights, or undersampling (check imbalance ratio from Phase 1)
- Record training time for each model

### 4.2 Deep Learning Models

- **Bi-directional LSTM**:
  - Input: tokenized and padded sequences from combined text
  - Embedding layer (pre-trained GloVe or trainable)
  - Bi-LSTM layers → Dense → Sigmoid output
  - Use validation set for early stopping
  - Hyperparameters: embedding dim, LSTM units, dropout, learning rate, batch size

- **DistilBERT**:
  - Fine-tune `distilbert-base-uncased` on the combined text
  - Use Hugging Face Trainer or manual training loop
  - Max sequence length: 512 tokens
  - Learning rate: 2e-5 to 5e-5 (standard fine-tuning range)
  - Epochs: 3-5 with early stopping

### 4.3 Hybrid Ensemble

- Combine best traditional ML model + Bi-LSTM + DistilBERT
- Methods:
  - Soft voting (average probabilities)
  - Stacking (meta-learner on base model predictions)
  - Weighted ensemble (tune weights on validation set)

**Deliverables**:

- Trained models saved (`.pkl`, `.h5`, `.pth` formats)
- Training logs and loss curves
- Hyperparameter configurations documented

**Tools**: Google Colab, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, XGBoost

---

## Phase 5: Model Evaluation

### 5.1 Performance Metrics

Evaluate each model on the held-out test set:

- **Accuracy**
- **Precision** (minimize false positives)
- **Recall** (minimize false negatives — critical for scam detection)
- **F1-score**
- **ROC-AUC**
- Confusion matrix
- Precision-Recall curve
- Classification report

### 5.2 Cross-Dataset Comparison

Compare performance between Dataset 1 and Dataset 2 models:

| Metric     | Dataset 1 Models | Dataset 2 Models | Notes              |
|------------|-------------------|-------------------|--------------------|
| Accuracy   | —                 | —                 |                    |
| Precision  | —                 | —                 |                    |
| Recall     | —                 | —                 | Priority metric    |
| F1-Score   | —                 | —                 |                    |
| ROC-AUC    | —                 | —                 |                    |

- Analyze whether the larger Dataset 2 (54k rows) yields better performance
- Analyze impact of missing structural features (no `has_company_logo`, `telecommuting`, etc.)
- Discuss whether the pre-encoded `location`/`industry` columns help or hinder

### 5.3 Error Analysis

- Examine false positives and false negatives
- Identify common patterns in misclassified posts
- Check if specific `employment_type` or `industry` categories are harder to classify
- Compare error patterns between Dataset 1 and Dataset 2

### 5.4 Interpretability

- SHAP values for feature importance (traditional ML models)
- LIME explanations for individual predictions
- Attention visualization for DistilBERT (optional)

**Deliverables**:

- Evaluation tables and visualizations in notebook
- Cross-dataset comparison table
- Error analysis with examples
- Best model selection with justification

**Tools**: Google Colab, scikit-learn metrics, SHAP, LIME, matplotlib/seaborn

---

## Phase 6: Dataset Merging Strategy (Future)

After both datasets are independently trained and evaluated, consider:

### 6.1 Column Harmonization

Map Dataset 2 columns to a unified schema:

```
Unified Schema:
- job_title       ← title (D1) / job_title (D2)
- job_desc        ← description (D1) / job_desc (D2)
- skills_desc     ← requirements (D1) / skills_desc (D2)
- company_profile ← company_profile (D1 & D2)
- salary_range    ← salary_range (D1 & D2)
- employment_type ← employment_type (D1 & D2)
- fraudulent      ← fraudulent (D1 & D2)
- combined_text   ← reconstructed from above fields
```

Columns that only exist in Dataset 1 (`telecommuting`, `has_company_logo`, `has_questions`, `department`, `benefits`, `required_experience`, `required_education`, `function`) would need to be dropped or filled with defaults when merging.

### 6.2 Combined Training

- Train on merged dataset with unified columns
- Compare merged model performance vs individual dataset models
- Evaluate generalization across data sources

---

## Notebook Structure

All implementation for Dataset 2 in a single notebook:

```
dataset2-scam-detection.ipynb
├── Section 1: Setup and Installation
│   ├── Import libraries
│   ├── Mount Google Drive (if needed)
│   └── Install required packages (openpyxl for .xlsx)
├── Section 2: Data Loading and EDA
│   ├── Load .xlsx file
│   ├── Column inspection and null analysis
│   ├── Label distribution (fraudulent column)
│   ├── Text column investigation
│   └── Pre-encoded column analysis (location, industry)
├── Section 3: Data Cleaning
│   ├── Drop unlabeled rows (fraudulent is NaN)
│   ├── Convert fraudulent to int
│   ├── Handle missing values per column
│   └── Remove duplicates
├── Section 4: Text Preprocessing
│   ├── HTML/URL removal
│   ├── Text normalization
│   ├── Combined text field creation/verification
│   └── Stop word removal and lemmatization
├── Section 5: Feature Engineering
│   ├── Text-based features (length, keyword, sentiment)
│   ├── Structural features (has_salary, has_profile, etc.)
│   ├── TF-IDF vectorization
│   └── Feature matrix construction
├── Section 6: Train/Val/Test Split
│   ├── Stratified split (70/15/15)
│   └── Class imbalance handling (SMOTE/class weights)
├── Section 7: Traditional ML Training
│   ├── Logistic Regression, Naive Bayes, SVM
│   ├── Decision Tree, Random Forest, XGBoost
│   └── Hyperparameter tuning
├── Section 8: Deep Learning Training
│   ├── Bi-LSTM
│   ├── DistilBERT fine-tuning
│   └── Model saving
├── Section 9: Hybrid Ensemble
│   ├── Voting / Stacking ensemble
│   └── Final hybrid model
├── Section 10: Evaluation
│   ├── Metrics (Accuracy, Precision, Recall, F1, AUC)
│   ├── Confusion matrices and ROC curves
│   ├── Cross-dataset comparison (vs Dataset 1 results)
│   └── Error analysis
├── Section 11: Interpretability
│   ├── SHAP / LIME analysis
│   └── Feature importance visualization
└── Section 12: Results and Conclusion
    ├── Summary of findings
    ├── Comparison with Dataset 1
    └── Recommendations for merging
```

---

## Key Differences from Implementation Plan 1

| Aspect                  | Plan 1 (Dataset 1)                     | Plan 2 (Dataset 2)                           |
|------------------------|----------------------------------------|-----------------------------------------------|
| File format            | CSV                                    | Excel (.xlsx) — requires openpyxl             |
| Dataset size           | 17,880 rows                            | 54,391 rows (44,045 labeled)                  |
| Label completeness     | 100% labeled                           | ~81% labeled — must drop/handle NaN labels    |
| Label dtype            | int64                                  | float64 — needs cast to int                   |
| Structural features    | 8 metadata columns                     | 3 metadata columns (fewer signals)            |
| location / industry    | Raw text (extractable features)        | Pre-encoded int64 (no text to analyze)        |
| Text column            | Must concatenate manually              | Pre-built `text` column available             |
| Primary text columns   | description, requirements              | job_desc, skills_desc                         |
| Feature engineering    | Rich structural + text features        | Heavier reliance on text-based features       |

## Success Metrics (Same Targets as Plan 1)

- Model accuracy: >85%
- Recall: >90% (critical for scam detection)
- F1-score: >85%
- System response time: <2 seconds per prediction
