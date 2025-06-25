import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random

# === 1. Load CSV Data ===
def load_dataset(csv_path):
    columns = ['id', 'text', 'sentiment']
    df = pd.read_csv(csv_path, sep=',', names=columns)
    return df.iloc[1:].reset_index(drop=True)

# === 2. Clean Text ===
def clean_text(text):
    if pd.isnull(text):
        text = ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip() if text.strip() else "link"

def apply_text_cleaning(df):
    df['text'] = df['text'].apply(clean_text)
    return df

# === 3. Balance Dataset ===
def balance_dataset(df, threshold=0.6):
    proportions = df['sentiment'].value_counts(normalize=True)
    majority = proportions.idxmax()
    minority = proportions.idxmin()

    if proportions[majority] > threshold:
        print(f"Most tweets are class {majority}")
        df_majority = df[df['sentiment'] == majority]
        df_minority = df[df['sentiment'] == minority]
        df_majority_reduced = df_majority.sample(n=len(df_minority), random_state=42)
        df_balanced = pd.concat([df_majority_reduced, df_minority]).sample(frac=1, random_state=42)
        df_remaining = df_majority.drop(df_majority_reduced.index)
        return df_balanced, df_remaining
    else:
        print("✅ Dataset already balanced.")
        return df.copy(), pd.DataFrame()

# === 4. TF-IDF Vectorization ===
def build_vectorizer(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectorizer.fit(texts)
    return vectorizer

# === 5. Split Data ===
def split_data(df_balanced, df_remaining):
    df_train_test, df_val_bal = train_test_split(
        df_balanced, test_size=0.25, stratify=df_balanced['sentiment'], random_state=42
    )
    df_val = pd.concat([df_val_bal, df_remaining]).sample(frac=1, random_state=42)
    return df_train_test, df_val

# === 6. Train Model with Cross-Validation ===
def train_model(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"F1-scores per fold: {scores}")
    print(f"Mean F1-score: {scores.mean():.4f}")
    model.fit(X_train, y_train)
    return model

# === 7. Evaluate Model ===
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("\n=== Evaluation on Validation Set ===")
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    return y_pred

# === 8. Plot Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not Depressed", "Depressed"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Validation Set")
    plt.tight_layout()
    plt.show()

# === 9. Save Model and Vectorizer ===
def save_model(model, vectorizer, output_path):
    joblib.dump((model, vectorizer), output_path)
    print(f"\nModel saved at: {output_path}")

# === MAIN EXECUTION FLOW ===
if __name__ == "__main__":
    x = random.randint(1, 1000000)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'tweets_csv', 'sentiment_tweets.csv')

    df = load_dataset(csv_path)
    df = apply_text_cleaning(df)
    df_balanced, df_remaining = balance_dataset(df)

    vectorizer = build_vectorizer(df_balanced['text'])
    df_train_test, df_validation = split_data(df_balanced, df_remaining)

    X_train_test = vectorizer.transform(df_train_test['text'])
    y_train_test = df_train_test['sentiment'].astype(int)
    X_validation = vectorizer.transform(df_validation['text'])
    y_validation = df_validation['sentiment'].astype(int)

    model = train_model(X_train_test, y_train_test)
    y_predicted = evaluate_model(model, X_validation, y_validation)
    plot_confusion_matrix(y_validation, y_predicted)

    model_output_path = os.path.join(base_dir, f"logreg_model_tweets{x}.joblib")
    save_model(model, vectorizer, model_output_path)
