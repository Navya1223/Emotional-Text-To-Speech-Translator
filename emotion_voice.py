"""Command-line version: type text, choose an emotion (or auto-detect), hear it spoken."""
import pickle

import pyttsx3
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 300
SENTIMENT_THRESHOLDS = (0.4, 0.7)
VOICE_SETTINGS = {
    "happy": {"rate": 150, "volume": 1.0},
    "neutral": {"rate": 120, "volume": 0.9},
    "sad": {"rate": 90, "volume": 0.6},
}

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model("model.h5")


def detect_emotion(text):
    x = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    score = float(model.predict(x, verbose=0)[0][0])
    if score <= SENTIMENT_THRESHOLDS[0]:
        return "sad"
    if score >= SENTIMENT_THRESHOLDS[1]:
        return "happy"
    return "neutral"


def speak(text, emotion=None, gender="female"):
    emotion = emotion or detect_emotion(text)
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    index = 1 if gender == "female" and len(voices) > 1 else 0  # usually 0 = male, 1 = female
    engine.setProperty("voice", voices[index].id)
    engine.setProperty("rate", VOICE_SETTINGS[emotion]["rate"])
    engine.setProperty("volume", VOICE_SETTINGS[emotion]["volume"])
    print(f"Speaking with emotion: {emotion}")
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    text = input("Enter your text: ")
    emotion = input("Emotion (happy, sad, neutral) or leave empty to auto-detect: ").strip().lower()
    gender = input("Voice (male, female): ").strip().lower() or "female"
    speak(text, emotion if emotion in VOICE_SETTINGS else None, gender)
