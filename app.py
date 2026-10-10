"""Emotional Text-to-Speech Translator (Streamlit app).

Type a sentence, choose a voice and an emotion (or let the model detect it),
and get back speech whose speed and loudness match the mood of the text.
"""
import os
import pickle
import tempfile

import pyttsx3
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 300
SENTIMENT_THRESHOLDS = (0.4, 0.7)  # score <= 0.4 -> sad, >= 0.7 -> happy, otherwise neutral

# Speaking rate (words per minute) and volume (0-1) for each emotion
VOICE_SETTINGS = {
    "happy": {"rate": 150, "volume": 1.0},
    "neutral": {"rate": 120, "volume": 0.9},
    "sad": {"rate": 90, "volume": 0.6},
}


@st.cache_resource
def load_model():
    """Load the trained LSTM model and tokenizer created by sentiment_modelling.ipynb."""
    if not (os.path.exists("model.h5") and os.path.exists("tokenizer.pkl")):
        return None, None
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model("model.h5")
    return tokenizer, model


def detect_emotion(text, tokenizer, model):
    """Return (emotion, score) where score is the model's positive-sentiment probability."""
    x = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = float(model.predict(x, verbose=0)[0][0])
    if score <= SENTIMENT_THRESHOLDS[0]:
        return "sad", score
    if score >= SENTIMENT_THRESHOLDS[1]:
        return "happy", score
    return "neutral", score


def pick_voice(engine, gender):
    """Choose a male or female voice from the voices installed on this computer."""
    voices = engine.getProperty("voices")
    for voice in voices:
        name = f"{voice.name} {voice.id}".lower()
        if gender == "female" and any(k in name for k in ("female", "zira", "+f")):
            return voice.id
        if gender == "male" and any(k in name for k in ("male", "david", "+m")) and "female" not in name:
            return voice.id
    # Fallback: on most systems voice 0 is male and voice 1 is female
    index = 1 if gender == "female" and len(voices) > 1 else 0
    return voices[index].id


def synthesize(text, emotion, gender):
    """Speak the text into a WAV file and return its bytes, so the browser can play it."""
    engine = pyttsx3.init()
    engine.setProperty("voice", pick_voice(engine, gender))
    engine.setProperty("rate", VOICE_SETTINGS[emotion]["rate"])
    engine.setProperty("volume", VOICE_SETTINGS[emotion]["volume"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "speech.wav")
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            return f.read()


st.set_page_config(page_title="Emotional Text-to-Speech", page_icon=":microphone:", layout="centered")
st.title("Emotional Text-to-Speech :microphone:")
st.caption("Type what you want to say. The app speaks it with a voice that matches the emotion.")

tokenizer, model = load_model()

text = st.text_area("Enter your text:", height=100)
col1, col2 = st.columns(2)
gender = col1.selectbox("Voice:", ["female", "male"])
choice = col2.selectbox("Emotion:", ["auto-detect", "happy", "sad", "neutral"])

if st.button("Convert to audio"):
    if not text.strip():
        st.warning("Please enter some text first.")
    elif choice == "auto-detect" and model is None:
        st.error("model.h5 and tokenizer.pkl not found. Run sentiment_modelling.ipynb first to create them.")
    else:
        if choice == "auto-detect":
            emotion, score = detect_emotion(text, tokenizer, model)
            st.info(f"Detected emotion: **{emotion}** (sentiment score {score:.2f})")
        else:
            emotion = choice
        audio = synthesize(text, emotion, gender)
        st.audio(audio, format="audio/wav")
        st.download_button("Download audio", audio, file_name=f"speech_{emotion}.wav", mime="audio/wav")

with st.expander("How to use"):
    st.markdown(
        """
1. Type your text.
2. Choose a female or male voice.
3. Choose an emotion, or leave **auto-detect** so the model decides from the text.
4. Click **Convert to audio** to play the speech, and download it if you want to share it.
"""
    )
