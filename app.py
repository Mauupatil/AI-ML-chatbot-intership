import streamlit as st
from retriever import retrieve
from deep_translator import GoogleTranslator
from textblob import TextBlob

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 Simple medical Q/A Chatbot")

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Marathi"]
)

# -------------------- TRANSLATION --------------------
def translate_to_english(text, lang):
    if lang == "English":
        return text
    return GoogleTranslator(source="auto", target="en").translate(text)

def translate_from_english(text, lang):
    if lang == "English":
        return text
    target_lang = "hi" if lang == "Hindi" else "mr"
    return GoogleTranslator(source="en", target=target_lang).translate(text)

# -------------------- SENTIMENT --------------------
def detect_sentiment(text):
    text_lower = text.lower()

    negative_keywords = [
        "worried", "afraid", "scared", "anxious",
        "stress", "depressed", "fear", "concern"
    ]
    positive_keywords = [
        "happy", "glad", "excited", "love"
    ]

    for word in negative_keywords:
        if word in text_lower:
            return "negative"

    for word in positive_keywords:
        if word in text_lower:
            return "positive"

    return "neutral"

# -------------------- MEDICAL CHECK --------------------
def is_medical_information_question(text):
    medical_keywords = [
        "diabetes", "fever", "symptoms", "treatment",
        "medicine", "disease", "blood sugar",
        "hypertension", "infection", "pain",
        "health", "condition"
    ]

    text_lower = text.lower()
    for word in medical_keywords:
        if word in text_lower:
            return True

    return False

# -------------------- UI --------------------
st.write("Ask a question based on the available data.")
question = st.text_input("Your question:")

if question:
    # 1️⃣ Translate question
    english_question = translate_to_english(question, language)

    # 2️⃣ Detect sentiment
    sentiment = detect_sentiment(english_question)

    # 3️⃣ Retrieve context
    contexts = retrieve(english_question)
    final_answer = " ".join(contexts)

    # 4️⃣ Apply sentiment ONLY if NOT medical
    
    if sentiment == "negative":
            final_answer = "I understand your concern. " + final_answer
    elif sentiment == "positive":
            final_answer = "Glad to help! " + final_answer

    # 5️⃣ Translate back
    if language != "English":
        translated_answer = translate_from_english(final_answer, language)
    else:
        translated_answer = final_answer

    st.subheader("Final Answer")
    st.write(translated_answer)