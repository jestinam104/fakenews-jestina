import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve
import joblib

# Optional: handle XGBoost
try:
    import xgboost as xgb
    xgb_installed = True
except ImportError:
    xgb_installed = False

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])

# Load data
@st.cache_data
def load_data():
    train = pd.read_csv("xy_train.csv")
    test = pd.read_csv("x_test.csv")
    if 'label' not in test.columns:
        test['label'] = 0
    return train, test

train, test = load_data()

# Clean text
train['clean_text'] = train['text'].apply(clean_text)
test['clean_text'] = test['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['clean_text'])
X_test = vectorizer.transform(test['clean_text'])

y_train = train['label']
y_test = test['label']

# Train models
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

if xgb_installed:
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

# Streamlit App
st.title("📰 Fake News Detection on Reddit Posts")
st.write("Enter a Reddit post and choose a model to detect whether it's fake or not.")

input_text = st.text_area("Enter the Reddit post text:")
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"] + (["XGBoost"] if xgb_installed else []))

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])
    
    if model_choice == "Logistic Regression":
        pred = lr_model.predict(vectorized)[0]
    elif model_choice == "Random Forest":
        pred = rf_model.predict(vectorized)[0]
    elif model_choice == "XGBoost":
        pred = xgb_model.predict(vectorized)[0]

    label = "🟢 Real" if pred == 0 else "🔴 Fake"
    st.subheader(f"Prediction: {label}")

# Display metrics
if st.checkbox("Show model evaluation metrics"):
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    st.write("### Logistic Regression Report")
    st.text(classification_report(y_test, lr_preds))
    st.write("### Random Forest Report")
    st.text(classification_report(y_test, rf_preds))

    if xgb_installed:
        xgb_preds = xgb_model.predict(X_test)
        st.write("### XGBoost Report")
        st.text(classification_report(y_test, xgb_preds))

# Plot ROC
if st.checkbox("Show ROC Curve"):
    import matplotlib.pyplot as plt

    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')

    if xgb_installed:
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
        plt.plot(fpr_xgb, tpr_xgb, label='XGBoost')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fake News Detection')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

