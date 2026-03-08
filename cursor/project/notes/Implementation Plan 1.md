# Scam Job Post Detection - Implementation Plan

## Overview

This plan outlines the implementation phases for building an AI system that detects scam job postings in the Philippine digital labor market using a hybrid learning approach (traditional ML + deep learning).

## Phase 1: Data Collection and Preprocessing

### 1.1 Dataset Collection

- **Kaggle Dataset**: Download and analyze the existing "Real or Fake Job Postings" dataset from Kaggle
  - Reference: https://www.kaggle.com/code/gauravsahani/real-or-fake-job-postings-with-bi-directional-lstm/notebook
  - Expected format: CSV with labeled job posts (fraudulent/legitimate)

- **Philippine Platform Data Collection**:
  - Scrape/collect job posts from Facebook Jobs (Philippine groups)
  - Collect from JobStreet Philippines
  - Gather from PinoyExchange forums
  - Manual collection from OnlineJobs.ph
  - Store in structured format (CSV/JSON)

- **Data Requirements**:
  - Minimum target: 5,000-10,000 labeled job posts
  - Balanced dataset (50/50 or 60/40 legitimate/scam ratio)
  - Include metadata: source platform, date, job title, description, company info, salary, location

### 1.2 Data Annotation and Labeling

- **Annotation Strategy**:
  - Manual labeling by domain experts (identify job market patterns)
  - Use annotation guidelines for English content
  - Measure inter-annotator agreement (Cohen's Kappa)
  - Create annotation guidelines document

- **Label Categories**:
  - Binary: `scam` (1) or `legitimate` (0)
  - Optional: confidence level, scam type (data theft, upfront payment, identity fraud)

### 1.3 Data Preprocessing

- **Text Cleaning**:
  - Remove duplicates
  - Handle special characters and encoding issues
  - Normalize whitespace
  - Remove irrelevant content (ads, navigation text)

- **Text Normalization**:
  - Lowercasing
  - Handle English text
  - Stemming/lemmatization
  - Remove stop words (English stop words)

- **Data Validation**:
  - Check for missing critical fields
  - Validate label consistency
  - Create train/validation/test splits (70/15/15)

**Deliverables**:

- Cleaned and labeled dataset (CSV/JSON) - saved in notebook or Google Drive
- Data preprocessing functions in notebook
- Dataset statistics and visualizations in notebook
- Annotation guidelines document (separate markdown file or notebook section)

**Tools**: Google Colab, Python, pandas, BeautifulSoup/Scrapy (for scraping), NLTK/spaCy

---

## Phase 2: Feature Engineering

### 2.1 Linguistic Features

- **Grammar and Language Quality**:
  - Grammar error detection (using language models)
  - Sentence structure analysis
  - Punctuation patterns (excessive exclamation marks, all caps)
  - Language detection (English)

- **Sentiment and Tone**:
  - Sentiment analysis (exaggerated positive sentiment)
  - Emotional tone detection
  - Second-person pronoun frequency ("you")

- **Keyword and Phrase Analysis**:
  - Scam indicator keywords: "easy money", "work from home", "no experience needed", "quick money", "earn thousands"
  - Scam phrases in English
  - Promotional language patterns

### 2.2 Structural Features

- **Job Post Structure**:
  - Post length (character count, word count)
  - Missing company details (company name, website, contact info)
  - Vague job titles
  - Missing job requirements/qualifications
  - Absence of application process details

- **Metadata Features**:
  - Source platform type
  - Presence of URLs/links (especially suspicious domains)
  - Contact method (WhatsApp, Telegram, email patterns)
  - Salary information (unrealistic amounts, missing salary)
  - Location information (vague locations, missing address)

### 2.3 Text Vectorization

- **Traditional Methods**:
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Bag-of-Words (BoW)
  - Character n-grams

- **Embedding Methods**:
  - Word2Vec (trained on job post corpus)
  - GloVe embeddings
  - English word embeddings

**Deliverables**:

- Feature extraction functions in notebook
- Feature importance visualizations
- Feature engineering code with comments
- Vectorized dataset (saved or kept in memory for training)

**Tools**: Google Colab, scikit-learn, NLTK, spaCy, transformers (Hugging Face), BERT for English text

---

## Phase 3: Model Selection and Training

### 3.1 Traditional Machine Learning Models

- **Baseline Models**:
  - Logistic Regression (interpretable baseline)
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (XGBoost)

- **Training Approach**:
  - Train on TF-IDF/BoW features
  - Hyperparameter tuning (GridSearchCV/RandomSearchCV)
  - Cross-validation (5-fold)
  - Handle class imbalance (SMOTE, class weights)

### 3.2 Deep Learning Models

Two deep learning models are used to cover distinct architectural families while keeping the study focused and reproducible:

- **Bi-directional LSTM** (sequential): Captures word order and long-range context in job descriptions; can use existing text embeddings (e.g. GloVe/Word2Vec from Phase 2). Represents the RNN/sequence family.
- **DistilBERT** (transformer): Pre-trained contextual model; lighter and faster than full BERT, suitable for inference speed targets (e.g. &lt;2 s). Represents the transformer family.

*Rationale*: One sequential + one transformer allows comparison of paradigms; prioritizes recall and deployment feasibility. Other options (unidirectional LSTM, CNN, CNN-LSTM, full BERT) are out of scope but may be noted as future work.

### 3.3 Hybrid Learning Approach

- **Ensemble Methods**:
  - Combine traditional ML (Random Forest, Logistic Regression) with deep learning (Bi-LSTM, DistilBERT)
  - Voting classifier (hard/soft voting)
  - Stacking ensemble
  - Weighted average of predictions

- **Model Selection Criteria**:
  - Balance between accuracy and interpretability
  - Consider inference speed for real-time deployment
  - Prioritize recall (minimize false negatives - missing scams)

**Deliverables**:

- Trained model files (pickle/ONNX/H5 formats) - saved to Google Drive or local
- Model training code in notebook with clear sections
- Hyperparameter configurations documented in notebook
- Model comparison tables and visualizations in notebook

**Tools**: Google Colab, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers

---

## Phase 4: Model Evaluation

### 4.1 Performance Metrics

- **Primary Metrics**:
  - Accuracy
  - Precision (minimize false positives)
  - Recall (minimize false negatives - critical for scam detection)
  - F1-score (harmonic mean)
  - ROC-AUC score

- **Additional Analysis**:
  - Confusion matrix
  - Precision-Recall curve
  - Feature importance analysis
  - Error analysis (false positives/negatives)

### 4.2 Model Validation

- **Cross-Validation**:
  - 5-fold cross-validation on training set
  - Stratified k-fold (maintain class distribution)

- **Test Set Evaluation**:
  - Final evaluation on held-out test set
  - Performance on different data sources (Facebook vs JobStreet)
  - Performance on English posts

### 4.3 Model Comparison and Selection

- **Comparison Matrix**:
  - Compare all models across metrics
  - Consider trade-offs (speed vs accuracy)
  - Select best-performing model(s) for deployment

- **Interpretability Analysis**:
  - SHAP values for feature importance
  - LIME explanations for individual predictions
  - Model decision visualization

**Deliverables**:

- Evaluation metrics and visualizations in notebook
- Model performance comparison table (in notebook)
- Best model selection with justification (documented in markdown cells)
- Error analysis with examples (in notebook)

**Tools**: Google Colab, scikit-learn metrics, SHAP, LIME, matplotlib/seaborn for visualization

---

## Phase 5: System Deployment and Interface Development

### 5.1 Backend Development

- **Model Serving**:
  - Save best model(s) in production-ready format
  - Create prediction API (Flask/FastAPI)
  - Implement preprocessing pipeline as API endpoint
  - Add confidence score calculation

- **API Endpoints**:
  - `POST /predict` - Single job post prediction
  - `POST /batch-predict` - Multiple job posts
  - `GET /health` - System health check
  - Return: prediction (scam/legitimate), confidence score, warning signals

### 5.2 Frontend Development (Web Tool)

- **Web Application**:
  - Technology: Streamlit (quick prototype) or Flask/Django + React
  - User interface:
    - Text input area for job post
    - Submit button
    - Results display:
      - Scam risk score (percentage)
      - Classification (Scam/Legitimate)
      - Warning signals (e.g., "Suspicious keywords detected", "Missing company details")
      - Color-coded risk indicator (red/yellow/green)
  - Additional features:
    - History of analyzed posts
    - Feedback mechanism (user can report misclassification)

### 5.3 Browser Extension (Optional)

- **Extension Development**:
  - Technology: JavaScript (Chrome Extension/Web Extension)
  - Functionality:
    - Detect job posts on Facebook, JobStreet pages
    - Auto-analyze and display risk badge
    - Click to see detailed analysis
  - Integration:
    - Call backend API for predictions
    - Store results locally (optional)

**Deliverables**:

- Deployed web application (local or cloud)
- API documentation
- Browser extension (if implemented)
- User guide

**Tools**: Flask/FastAPI, Streamlit, React (optional), JavaScript (for extension), Docker (for containerization)

---

## Phase 6: System Testing and Feedback

### 6.1 System Testing

- **Functional Testing**:
  - Test prediction accuracy on new job posts
  - Test API endpoints
  - Test user interface usability
  - Test browser extension (if implemented)

- **Performance Testing**:
  - Response time (target: <2 seconds per prediction)
  - Load testing (concurrent users)
  - Model inference speed

### 6.2 User Testing

- **Beta Testing**:
  - Recruit Filipino job seekers (students, freelancers, first-time applicants)
  - Collect feedback on:
    - Usability
    - Accuracy of predictions
    - Trust in the system
    - Interface clarity

- **Feedback Collection**:
  - User surveys
  - Misclassification reports
  - Feature requests

### 6.3 Model Refinement

- **Continuous Improvement**:
  - Retrain model with user feedback data
  - Update feature engineering based on new scam patterns
  - A/B testing of different models
  - Monitor performance over time

**Deliverables**:

- Testing report
- User feedback summary
- Refined model (if improvements made)
- Final system documentation

**Tools**: pytest (for testing), user survey tools, monitoring tools

---

## Project Structure

### Single Notebook Approach (Google Colab)

All implementation will be done in a single Jupyter/Colab notebook organized into clear sections:

```
scam-job-detection-notebook.ipynb
├── Section 1: Setup and Installation
│   ├── Import libraries
│   ├── Mount Google Drive (if needed)
│   └── Install required packages
├── Section 2: Data Collection and Loading
│   ├── Load Kaggle dataset
│   ├── Load Philippine platform data
│   └── Data overview and statistics
├── Section 3: Data Preprocessing
│   ├── Text cleaning functions
│   ├── Text normalization
│   ├── Handle English content
│   └── Train/validation/test split
├── Section 4: Feature Engineering
│   ├── Linguistic features extraction
│   ├── Structural features extraction
│   ├── Text vectorization (TF-IDF, embeddings)
│   └── Feature combination and selection
├── Section 5: Model Training - Traditional ML
│   ├── Logistic Regression
│   ├── Random Forest
│   ├── SVM
│   ├── XGBoost
│   └── Hyperparameter tuning
├── Section 6: Model Training - Deep Learning
│   ├── Bi-LSTM
│   ├── DistilBERT fine-tuning
│   └── Model saving
├── Section 7: Hybrid Ensemble
│   ├── Combine traditional ML and deep learning
│   ├── Voting/Stacking ensemble
│   └── Final hybrid model
├── Section 8: Model Evaluation
│   ├── Performance metrics calculation
│   ├── Confusion matrices
│   ├── ROC curves
│   ├── Model comparison
│   └── Error analysis
├── Section 9: Model Interpretation
│   ├── Feature importance (SHAP/LIME)
│   └── Visualization of predictions
├── Section 10: Deployment Preparation
│   ├── Save final model
│   ├── Create prediction function
│   └── Export for web deployment
└── Section 11: Results and Conclusion
    ├── Summary of findings
    ├── Best model selection
    └── Future improvements

Supporting Files (in Google Drive or local):
├── data/
│   ├── raw/              # Raw collected data
│   ├── processed/        # Cleaned datasets
│   └── labeled/          # Final labeled dataset
├── models/               # Saved model files (.pkl, .h5, .pth)
└── requirements.txt      # Package dependencies
```

**Notebook Organization Tips**:

- Use markdown cells for section headers and documentation
- Group related code cells together
- Add clear comments and explanations
- Use markdown cells to separate major phases
- Save intermediate results to avoid re-running expensive operations
- Use `%%time` magic commands to track execution time

## Key Considerations

1. **Context**: Handle English job posts, local platforms, cultural nuances
2. **Class Imbalance**: Use techniques like SMOTE, class weights, or oversampling
3. **Interpretability**: Balance accuracy with explainability for user trust
4. **Real-time Performance**: Optimize for fast inference in production
5. **Ethical Considerations**: Ensure fair detection, avoid bias against legitimate posts
6. **Data Privacy**: Handle user-submitted job posts securely

## Success Metrics

- Model accuracy: >85%
- Recall: >90% (critical for scam detection)
- User satisfaction: >80% positive feedback
- System response time: <2 seconds
- Deployment: Functional web tool accessible to Filipino job seekers
