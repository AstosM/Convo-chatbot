import streamlit as st
import json, os, datetime
import pandas as pd
import numpy as np
import nltk

# download required NLTK resources
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- CONFIG ----------
INTENTS_FILE = "intents.json"
LOG_FILE = "chat_logs.csv"
CONF_THRESHOLD = 0.55
lemmatizer = WordNetLemmatizer()

# ---------- HELPERS ----------
def normalize(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if any(c.isalnum() for c in t)]
    return " ".join(tokens)

def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]

def log(user, bot, intent, conf, mode):
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "user_text": user,
        "bot_text": bot,
        "intent": intent,
        "confidence": conf,
        "mode": mode
    }
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# ---------- LOAD DATA ----------
intents = load_intents()

X, y = [], []
for intent in intents:
    for lang, examples in intent["examples"].items():
        for ex in examples:
            X.append(normalize(ex))
            y.append(intent["id"])

clf_vec = TfidfVectorizer()
Xv = clf_vec.fit_transform(X)
clf = LogisticRegression(max_iter=300)
clf.fit(Xv, y)

# ---------- STREAMLIT APP ----------
st.title("🎓 Multilingual Campus Chatbot")
st.caption("Ask about fees, scholarships, timetable… in English or Hindi.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("💬 Your question:")

if st.button("Ask") and user_input.strip():
    xv = clf_vec.transform([normalize(user_input)])
    probs = clf.predict_proba(xv)[0]
    pred = clf.classes_[np.argmax(probs)]
    conf = max(probs)

    bot_text, mode = None, None

    if conf >= CONF_THRESHOLD:
        intent_obj = next(it for it in intents if it["id"] == pred)
        bot_text = intent_obj["response"]["en"]
        mode = "intent"
    else:
        # keyword fallback
        if "fee" in user_input.lower() or "fess" in user_input.lower() or "fees" in user_input.lower():
            fee_intent = next(it for it in intents if it["id"] == "fee_info")
            bot_text = fee_intent["response"]["en"]
            mode = "keyword"

    # final fallback
    if not bot_text:
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

# safe log viewer
with st.expander("📑 Show recent logs"):
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            df = pd.read_csv(LOG_FILE)
            if df.shape[0] > 0:
                st.dataframe(df.tail(10))
            else:
                st.info("Log file exists but is empty.")
        except pd.errors.EmptyDataError:
            st.info("Log file is empty. No conversations yet.")
        except Exception as e:
            st.warning(f"⚠️ Could not read logs: {e}")
    else:
        st.info("No logs yet.")
