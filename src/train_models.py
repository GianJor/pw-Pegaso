import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# 1️⃣ CONFIGURAZIONE PERCORSI AUTOMATICA
# --------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "reviews_synth.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------------
# 2️⃣ FUNZIONE DI PULIZIA DEL TESTO
# --------------------------------------------------------------
def clean_text(text):
    """Pulisce il testo in modo coerente con l’app Streamlit."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Rimuove punteggiatura
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------------------------------------------
# 3️⃣ CARICAMENTO E PRE-PROCESSING DEL DATASET
# --------------------------------------------------------------
print(f"\n📂 Caricamento dataset da '{DATA_PATH}'...")
df = pd.read_csv(DATA_PATH)

df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)

print(f"➡️ Numero di recensioni: {len(df)}")
print(f"➡️ Classi reparto: {df['department'].unique()}")
print(f"➡️ Classi sentiment: {df['sentiment'].unique()}")

# --------------------------------------------------------------
# 4️⃣ SUDDIVISIONE TRAIN/TEST
# --------------------------------------------------------------
# Reparto
X_train, X_test, y_train_dep, y_test_dep = train_test_split(
    df["text"], df["department"], test_size=0.2, random_state=42, stratify=df["department"]
)

# Sentiment (attenzione: variabili separate!)
X_train_s, X_test_s, y_train_sent, y_test_sent = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

print(f"➡️ Training set Reparto: {len(X_train)} | Test set: {len(X_test)}")
print(f"➡️ Training set Sentiment: {len(X_train_s)} | Test set: {len(X_test_s)}")

# --------------------------------------------------------------
# 5️⃣ TF-IDF VECTORIZER
# --------------------------------------------------------------
print("\n🧠 Creazione vettorizzatore TF-IDF...")
vectorizer = TfidfVectorizer(lowercase=True, max_features=1500, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ⚠️ Creiamo anche i vettori separati per il modello Sentiment
X_train_vec_s = vectorizer.transform(X_train_s)
X_test_vec_s = vectorizer.transform(X_test_s)

print(f"➡️ TF-IDF dimensione: {X_train_vec.shape}")

# --------------------------------------------------------------
# 6️⃣ MODELLI DI CLASSIFICAZIONE
# --------------------------------------------------------------
def train_and_evaluate_model(X_train, X_test, y_train, y_test, name):
    """Addestra e valuta un modello di regressione logistica."""
    print(f"\n🚀 Addestramento modello: {name}")
    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"✅ {name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    print("\nReport dettagliato:\n", classification_report(y_test, y_pred))

    # Matrice di confusione
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"confusion_{name}.png"))
    plt.close()

    return model, acc, f1

# --------------------------------------------------------------
# 7️⃣ ADDESTRAMENTO MODELLI
# --------------------------------------------------------------
clf_dep, acc_dep, f1_dep = train_and_evaluate_model(
    X_train_vec, X_test_vec, y_train_dep, y_test_dep, "department"
)

clf_sent, acc_sent, f1_sent = train_and_evaluate_model(
    X_train_vec_s, X_test_vec_s, y_train_sent, y_test_sent, "sentiment"
)

# --------------------------------------------------------------
# 8️⃣ SALVATAGGIO MODELLI
# --------------------------------------------------------------
print(f"\n💾 Salvataggio modelli e vettorizzatore in '{MODEL_DIR}'...")
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
joblib.dump(clf_dep, os.path.join(MODEL_DIR, "department_model.joblib"))
joblib.dump(clf_sent, os.path.join(MODEL_DIR, "sentiment_model.joblib"))

print("\n🏁 Addestramento completato con successo!")
print(f"➡️ Department — Accuracy: {acc_dep:.3f}, F1: {f1_dep:.3f}")
print(f"➡️ Sentiment  — Accuracy: {acc_sent:.3f}, F1: {f1_sent:.3f}")
