import streamlit as st
import json, os, datetime
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# optional translation
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATE = True
except:
    translator = None
    TRANSLATE = False

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# ----------------- CONFIG -----------------
INTENTS_FILE = "intents.json"
LOG_FILE = "chat_logs.csv"
CONF_THRESHOLD = 0.55

# ----------------- HELPERS -----------------
def normalize(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if any(c.isalnum() for c in t)]
    return " ".join(tokens)

def translate_to_en(text: str) -> str:
    if TRANSLATE:
        try:
            return translator.translate(text, dest="en").text
        except:
            return text
    return text

def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]

def prepare_data(intents):
    X, y = [], []
    for intent in intents:
        for lang, examples in intent["examples"].items():
            for ex in examples:
                ex_en = translate_to_en(ex) if lang != "en" else ex
                X.append(normalize(ex_en))
                y.append(intent["id"])
    return X, y

def train_classifier(X, y):
    vec = TfidfVectorizer()
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=300)
    clf.fit(Xv, y)
    return vec, clf

def retrieve(text, vectorizer, doc_vectors, doc_meta, k=2):
    q = normalize(translate_to_en(text))
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, doc_vectors)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [(doc_meta[i], sims[i]) for i in top_idx]

def log(user, bot, intent, conf, mode):
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user_text": user,
        "bot_text": bot,
        "intent": intent,
        "confidence": conf,
        "mode": mode,
    }
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# ----------------- BUILD MODELS -----------------
intents = load_intents()
X, y = prepare_data(intents)
clf_vec, clf = train_classifier(X, y)

# retrieval corpus
docs = [normalize(translate_to_en(it["response"].get("en", ""))) for it in intents]
retr_vec = TfidfVectorizer()
doc_vecs = retr_vec.fit_transform(docs)
doc_meta = intents

# ----------------- STREAMLIT APP -----------------
st.title("🎓 Multilingual Campus Chatbot")
st.caption("Ask about fees, scholarships, timetable… in English or Hindi. Logs saved for continuous improvement.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("💬 Your question:", "")

if st.button("Ask") and user_input.strip():
    # predict
    txt_en = translate_to_en(user_input)
    xv = clf_vec.transform([normalize(txt_en)])
    probs = clf.predict_proba(xv)[0]
    pred = clf.classes_[np.argmax(probs)]
    conf = max(probs)

    # retrieval fallback
    retrievals = retrieve(user_input, retr_vec, doc_vecs, doc_meta)
    if conf >= CONF_THRESHOLD:
        intent_obj = next(it for it in intents if it["id"] == pred)
        bot_text = intent_obj["response"]["en"]
        mode = "intent"
    elif retrievals and retrievals[0][1] > 0.3:
        bot_text = retrievals[0][0]["response"]["en"]
        mode = "retrieval"
    else:
        bot_text = "⚠️ I couldn’t find a clear answer. Please contact the helpdesk."
        mode = "fallback"

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_text))

    log(user_input, bot_text, pred, conf, mode)

# show conversation
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.markdown(f"<span style='color:blue'>**{speaker}:** {text}</span>", unsafe_allow_html=True)

with st.expander("📑 Show last 10 logs"):
    if os.path.exists(LOG_FILE):
        st.dataframe(pd.read_csv(LOG_FILE).tail(10))
    else:
        st.info("No logs yet.")
