#ML development by KiruthikaVM2005
import bz2
import pandas as pd

train_file = "data/train.ft.txt.bz2"

data = []

with bz2.open(train_file, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        label, text = line.strip().split(" ", 1)

        data.append({
            "label": label,
            "text": text
        })

        if i == 99999:
            break

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print(df.head())
print(df["label"].value_counts())
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

print(df[["text", "clean_text"]].head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

X = vectorizer.fit_transform(df["clean_text"])

print("TF-IDF shape:", X.shape)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

print("Classes:", encoder.classes_)
print("Label shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

f1 = f1_score(y_val, y_pred, average="macro")

print("Logistic Regression Macro-F1:", f1)
from sklearn.svm import LinearSVC

svm_model = LinearSVC(
    class_weight="balanced"
)

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_val)

svm_f1 = f1_score(
    y_val,
    svm_pred,
    average="macro"
)

print("Linear SVM Macro-F1:", svm_f1)
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_val)

nb_f1 = f1_score(
    y_val,
    nb_pred,
    average="macro"
)

print("Multinomial Naive Bayes Macro-F1:", nb_f1)
tuned_model = LogisticRegression(
    C=2,
    max_iter=1000,
    class_weight="balanced"
)

tuned_model.fit(X_train, y_train)

tuned_pred = tuned_model.predict(X_val)

tuned_f1 = f1_score(
    y_val,
    tuned_pred,
    average="macro"
)

print("Tuned Logistic Regression Macro-F1:", tuned_f1)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_val, tuned_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_val, tuned_pred))
import joblib

joblib.dump(tuned_model, "ml/tuned_logistic_model.pkl")
joblib.dump(vectorizer, "ml/tfidf_vectorizer.pkl")

print("Model and TF-IDF vectorizer saved successfully!")
# Load test data
test_data = []

with bz2.open("data/test.ft.txt.bz2", "rt", encoding="utf-8") as f:
    for line in f:
        label, text = line.strip().split(" ", 1)

        test_data.append({
            "label": label,
            "text": text
        })

test_df = pd.DataFrame(test_data)

print("Test dataset shape:", test_df.shape)

# Load saved model and vectorizer
logistic_model = joblib.load("ml/tuned_logistic_model.pkl")
vectorizer = joblib.load("ml/tfidf_vectorizer.pkl")

# Prediction
X_test = vectorizer.transform(test_df["text"])

test_predictions = logistic_model.predict(X_test)

print("Number of predictions:", len(test_predictions))
print("First 10 predictions:", test_predictions[:10])
# Save predictions to CSV
prediction_df = pd.DataFrame({
    "prediction": test_predictions
})

prediction_df.to_csv("ml/test_predictions.csv", index=False)

print("Predictions saved successfully!")
