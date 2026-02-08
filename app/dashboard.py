import streamlit as st
import pandas as pd
import joblib
import datetime
import docx2txt
import re
from PyPDF2 import PdfReader


# ==============================================================
# CONFIGURAZIONE BASE STREAMLIT E STILI GRAFICI
# ==============================================================
st.set_page_config(page_title="Hotel Review Classifier", page_icon="🏨", layout="wide")

# Costante per la gestione dell'Off-Topic (se la probabilità è inferiore, si classifica come ambiguo)
CONFIDENCE_THRESHOLD = 0.25

# Stili CSS personalizzati per rendere l’interfaccia più moderna
st.markdown("""
<style>
.main > div { padding-top: 1rem; }
.section { background: #ffffff; border: 1px solid #eee; border-radius: 16px; padding: 1.2rem 1.2rem; }
.badge { display: inline-block; padding: .35rem .6rem; border-radius: 999px; font-weight: 600; border: 1px solid #e5e7eb; }
.badge-ok { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.badge-warn { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.card { border: 1px solid #eee; border-radius: 14px; padding: .9rem 1rem; }
.prob { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================
# FUNZIONI DI PRE-PROCESSING E UTILITY
# ==============================================================

def clean_text(text):
    """Semplice funzione di pulizia (deve essere consistente con il training)."""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Rimuove punteggiatura e caratteri speciali
    text = re.sub(r'\s+', ' ', text).strip() # Sostituisce spazi multipli
    return text

def predict_review(text, vectorizer, clf_dep, clf_sent):
    """Esegue la predizione di reparto e sentiment su un singolo testo pulito."""
    X = vectorizer.transform([text])
    
    # Reparto
    dep_pred = clf_dep.predict(X)[0]
    dep_probs = clf_dep.predict_proba(X)[0]
    dep_labels = clf_dep.classes_
    dep_df = pd.DataFrame({"Reparto": dep_labels, "Prob": dep_probs}).sort_values("Prob", ascending=False)
    
    # Sentiment
    sent_pred = clf_sent.predict(X)[0]
    sent_probs = clf_sent.predict_proba(X)[0]
    sent_labels = clf_sent.classes_
    sent_df = pd.DataFrame({"Sentiment": sent_labels, "Prob": sent_probs}).sort_values("Prob", ascending=False)
    
    return dep_pred, dep_df, sent_pred, sent_df

# ==============================================================
# CARICAMENTO MODELLI MACHINE LEARNING
# ==============================================================
@st.cache_resource
def load_artifacts():
    """Carica i modelli ML e il vettorizzatore TF-IDF salvati in fase di training."""
    try:
        vec = joblib.load("models/vectorizer.joblib")
        dep = joblib.load("models/department_model.joblib")
        sen = joblib.load("models/sentiment_model.joblib")
        return vec, dep, sen
    except FileNotFoundError:
        st.error("Errore: Impossibile trovare i file dei modelli in 'models/'. Assicurati di aver eseguito lo script di training.")
        st.stop()


vectorizer, clf_dep, clf_sent = load_artifacts()

# Dizionario per convertire i codici reparto in nomi estesi e leggibili
DEPARTMENT_LABELS = {
    "Housekeeping": "🧹 Servizio Pulizie",
    "F&B": "🍽️ Ristorazione (Food & Beverage)",
    "Reception": "🛎️ Reception / Accoglienza"
}


# ==============================================================
# HEADER E SIDEBAR
# ==============================================================
c1, c2 = st.columns([1, 3])
with c1:
    st.image("https://img.icons8.com/emoji/96/hotel-emoji.png", width=64)
with c2:
    st.markdown("## Hotel Review Classifier")
    st.caption(f"Smistamento automatico **Reparto** + **Sentiment** per recensioni hotel. Soglia di confidenza: **{int(CONFIDENCE_THRESHOLD*100)}%**.")

# Sidebar informativa con stato modelli e istruzioni
with st.sidebar:
    st.markdown("### ⚙️ Modelli")
    st.markdown('<span class="badge badge-ok">Caricati</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Suggerimenti input**")
    st.write("- Descrivi *pulizia*, *check-in/out*, *colazione/ristorante*")
    st.write("- Puoi lasciare il titolo vuoto")
    st.caption("Dataset sintetico • Logistic Regression")


# ==============================================================
# SEZIONE 1: ANALISI DI UNA RECENSIONE SINGOLA
# ==============================================================
st.markdown("#### ✍️ Analisi singola")

with st.container():
    colA, colB = st.columns([1, 2])

    # --- Colonna sinistra: input utente ---
    with colA:
        title = st.text_input("Titolo recensione", placeholder="Es. Ottimo soggiorno")
        body = st.text_area("Testo recensione", height=140, placeholder="Es. Camera pulita, check-in veloce e colazione super.")
        run = st.button("Analizza recensione", type="primary", use_container_width=True)

    # --- Colonna destra: risultati ---
    with colB:
        if run:
            raw_text = (title or "") + " " + body
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                st.warning("Inserisci testo valido (non solo spazi o caratteri speciali).")
                st.stop()
            
            if len(cleaned_text.split()) < 3:
                st.warning("Il testo è troppo breve (< 3 parole) per una classificazione affidabile.")
                st.stop()
            
            dep_pred, dep_df, sent_pred, sent_df = predict_review(cleaned_text, vectorizer, clf_dep, clf_sent)

            # --- Visualizzazione risultati ---
            st.markdown("##### Risultato")
            cR1, cR2 = st.columns(2)
            
            # Reparto
            max_dep_prob = float(dep_df.iloc[0]["Prob"])
            
            # Determina se il Reparto è Off-Topic
            is_off_topic = max_dep_prob < CONFIDENCE_THRESHOLD
            
            with cR1:
                st.markdown("###### Reparto")
                if is_off_topic:
                    dep_display = "❓ Non Classificabile / Off-Topic"
                    st.markdown(f'<div class="card bg-red-100">🏷️ **Reparto suggerito:** <span class="prob" style="color: #ef4444;">{dep_display}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob, text=f"Confidenza bassa: {max_dep_prob:.2f}")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                else:
                    dep_display = DEPARTMENT_LABELS.get(dep_pred, dep_pred)
                    st.markdown(f'<div class="card">🏷️ **Reparto suggerito:** <span class="prob">{dep_display}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob)
                    
                    dep_df["Reparto"] = dep_df["Reparto"].map(DEPARTMENT_LABELS)
                    st.dataframe(dep_df.reset_index(drop=True), hide_index=True, use_container_width=True) # MOSTRATO SOLO SE CLASSIFICATO

            # Sentiment
            sent_pred_prob = float(sent_df.loc[sent_df["Sentiment"] == sent_pred, "Prob"].values[0])
            
            with cR2:
                st.markdown("###### Sentiment")
                # Se il reparto è Off-Topic, anche il Sentiment deve essere Off-Topic
                if is_off_topic: 
                    emo = "❔ Non Classificabile"
                    # Usiamo la probabilità del reparto, che era bassa, come indicatore
                    st.markdown(f'<div class="card bg-red-100">🫶 **Sentiment stimato:** <span class="prob" style="color: #ef4444;">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob, text=f"Dipende da Reparto ({max_dep_prob:.2f})")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                elif sent_pred_prob < CONFIDENCE_THRESHOLD:
                    # Se il Reparto è OK ma il Sentiment è ambiguo (raro)
                    emo = "❔ Non Classificabile"
                    st.markdown(f'<div class="card bg-red-100">🫶 **Sentiment stimato:** <span class="prob" style="color: #ef4444;">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(sent_pred_prob, text=f"Confidenza bassa: {sent_pred_prob:.2f}")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                else:
                    # Risultato Sentiment valido
                    emo = "😊 positivo" if sent_pred == "pos" else "😞 negativo"
                    st.markdown(f'<div class="card">🫶 **Sentiment stimato:** <span class="prob">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(sent_pred_prob)
                    
                    st.dataframe(sent_df.reset_index(drop=True), hide_index=True, use_container_width=True) # MOSTRATO SOLO SE CLASSIFICATO


# ==============================================================
# SEZIONE 2: ANALISI IN BATCH (UPLOAD FILE CSV)
# ==============================================================
st.markdown("#### 📂 Analisi da file CSV")
st.caption("Il CSV deve avere **almeno** le colonne: `title`, `body`.")

uploaded = st.file_uploader("Carica CSV", type=["csv"])

if uploaded is not None:
    try:
        # Lettura sicura del CSV
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Impossibile leggere il file: {e}")
        st.stop()

    # Verifica struttura colonne
    missing = [c for c in ["title", "body"] if c not in df_in.columns]
    if missing:
        st.error(f"Mancano le colonne richieste: {missing}")
        st.stop()

    # Prepara i dati per la predizione e pulizia
    df_in = df_in.copy()
    df_in["title"] = df_in["title"].fillna("")
    df_in["body"] = df_in["body"].fillna("")
    df_in["raw_text"] = df_in["title"] + " " + df_in["body"]
    df_in["text"] = df_in["raw_text"].apply(clean_text) # Applico la pulizia

    # Rimuovo le recensioni troppo brevi o vuote dopo la pulizia
    df_in = df_in[df_in["text"].str.split().str.len() >= 3].reset_index(drop=True)
    
    if df_in.empty:
        st.error("Nessuna recensione valida trovata nel file dopo la pulizia e il filtro per lunghezza minima (3 parole).")
        st.stop()
        
    st.success(f"Analisi su {len(df_in)} recensioni valide in corso...")

    # Predizioni batch
    Xb = vectorizer.transform(df_in["text"])
    
    # 1. Reparto (e confidenza)
    dep_probs = clf_dep.predict_proba(Xb)
    df_in["pred_department_raw"] = clf_dep.predict(Xb)
    df_in["dep_max_prob"] = [max(p) for p in dep_probs]
    
    # Classificazione Off-Topic per il Reparto
    df_in["pred_department"] = df_in.apply(
        lambda row: DEPARTMENT_LABELS[row["pred_department_raw"]] if row["dep_max_prob"] >= CONFIDENCE_THRESHOLD else "🚫 Non Classificabile",
        axis=1
    )

    # 2. Sentiment (e confidenza)
    sent_probs = clf_sent.predict_proba(Xb)
    df_in["pred_sentiment_raw"] = clf_sent.predict(Xb)
    df_in["sent_max_prob"] = [max(p) for p in sent_probs]
    
    # Classificazione Off-Topic per il Sentiment (DIPENDE DALLA CONFIDENZA DEL REPARTO)
    def classify_sentiment_batch(row):
        # Se il reparto è Off-Topic, anche il sentiment è Off-Topic
        if row["pred_department"] == "🚫 Non Classificabile":
            return "❔ Non Classificabile (Off-Topic Reparto)"
        
        # Altrimenti, valuta solo la confidenza del sentiment
        if row["sent_max_prob"] >= CONFIDENCE_THRESHOLD:
            return row["pred_sentiment_raw"].replace("pos", "😊 positivo").replace("neg", "😞 negativo")
        else:
            return "❔ Non Classificabile"
            
    df_in["pred_sentiment"] = df_in.apply(classify_sentiment_batch, axis=1)
    
    # Output tabellare e download
    st.success("Predizioni completate ✅")
    
    # Visualizzo solo colonne rilevanti e i risultati finali
    st.dataframe(
        df_in[["title", "body", "pred_department", "dep_max_prob", "pred_sentiment", "sent_max_prob"]], 
        use_container_width=True,
        column_config={
            "dep_max_prob": st.column_config.ProgressColumn("Confidenza Reparto", format="%.2f", min_value=0, max_value=1),
            "sent_max_prob": st.column_config.ProgressColumn("Confidenza Sentiment", format="%.2f", min_value=0, max_value=1),
        }
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Scarica risultati (CSV)",
        data=out_csv,
        file_name=f"predictions_{ts}.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==============================================================
# SEZIONE 3: ANALISI DI FILE PDF O WORD
# ==============================================================
st.markdown("---")
st.subheader("📄 Carica un file PDF o Word")

uploaded_doc = st.file_uploader("Seleziona un file PDF o DOCX", type=["pdf", "docx"])

if uploaded_doc is not None:
    text = ""
    if uploaded_doc.name.endswith(".pdf"):
        reader = PdfReader(uploaded_doc)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_doc.name.endswith(".docx"):
        text = docx2txt.process(uploaded_doc)

    cleaned_text = clean_text(text)
    
    if cleaned_text:
        st.text_area("Testo estratto e pulito (primi 1000 caratteri)", cleaned_text[:1000])

        if st.button("Analizza documento", type="primary", use_container_width=True):
            if len(cleaned_text.split()) < 3:
                st.warning("Il testo estratto è troppo breve (< 3 parole) per una classificazione affidabile.")
                st.stop()
            
            dep_pred, dep_df, sent_pred, sent_df = predict_review(cleaned_text, vectorizer, clf_dep, clf_sent)

            # --- Visualizzazione risultati (Stessa logica di confidenza) ---
            st.markdown("##### Risultato")
            cR1, cR2 = st.columns(2)
            
            # Reparto
            max_dep_prob = float(dep_df.iloc[0]["Prob"])
            
            # Determina se il Reparto è Off-Topic
            is_off_topic = max_dep_prob < CONFIDENCE_THRESHOLD
            
            with cR1:
                st.markdown("###### Reparto")
                if is_off_topic:
                    dep_display = "❓ Non Classificabile / Off-Topic"
                    st.markdown(f'<div class="card bg-red-100">🏷️ **Reparto suggerito:** <span class="prob" style="color: #ef4444;">{dep_display}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob, text=f"Confidenza bassa: {max_dep_prob:.2f}")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                else:
                    dep_display = DEPARTMENT_LABELS.get(dep_pred, dep_pred)
                    st.markdown(f'<div class="card">🏷️ **Reparto suggerito:** <span class="prob">{dep_display}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob)
                    
                    dep_df["Reparto"] = dep_df["Reparto"].map(DEPARTMENT_LABELS)
                    st.dataframe(dep_df.reset_index(drop=True), hide_index=True, use_container_width=True) # MOSTRATO SOLO SE CLASSIFICATO

            # Sentiment
            sent_pred_prob = float(sent_df.loc[sent_df["Sentiment"] == sent_pred, "Prob"].values[0])
            
            with cR2:
                st.markdown("###### Sentiment")
                # Se il reparto è Off-Topic, anche il Sentiment deve essere Off-Topic
                if is_off_topic: # <-- NUOVA LOGICA DI DIPENDENZA
                    emo = "❔ Non Classificabile"
                    st.markdown(f'<div class="card bg-red-100">🫶 **Sentiment stimato:** <span class="prob" style="color: #ef4444;">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(max_dep_prob, text=f"Dipende da Reparto ({max_dep_prob:.2f})")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                elif sent_pred_prob < CONFIDENCE_THRESHOLD:
                    # Se il Reparto è OK ma il Sentiment è ambiguo
                    emo = "❔ Non Classificabile"
                    st.markdown(f'<div class="card bg-red-100">🫶 **Sentiment stimato:** <span class="prob" style="color: #ef4444;">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(sent_pred_prob, text=f"Confidenza bassa: {sent_pred_prob:.2f}")
                    # NON MOSTRIAMO IL DATAFRAME IN CASO DI OFF-TOPIC
                else:
                    # Risultato Sentiment valido
                    emo = "😊 positivo" if sent_pred == "pos" else "😞 negativo"
                    st.markdown(f'<div class="card">🫶 **Sentiment stimato:** <span class="prob">{emo}</span></div>', unsafe_allow_html=True)
                    st.progress(sent_pred_prob)
                    
                    st.dataframe(sent_df.reset_index(drop=True), hide_index=True, use_container_width=True) # MOSTRATO SOLO SE CLASSIFICATO
    else:
        st.error("Non è stato possibile estrarre testo valido dal file.")