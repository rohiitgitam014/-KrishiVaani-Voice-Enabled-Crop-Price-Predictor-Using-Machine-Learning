import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import pandas as pd
import re
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="🌾🔊 KrishiVaani: Voice-Enabled Crop Price Predictor Using Machine Learning", layout="centered")
st.title("🎤 Record or Upload Audio to Predict Crop Price (🌍 Multilingual)")

# Record or upload
st.subheader("🎙️ Step 1: Record or Upload Audio")
audio_bytes = audio_recorder(text="Click to Record", recording_color="#e8b62c", neutral_color="#6aa36f", icon_size="2x")
uploaded_file = st.file_uploader("Or upload a .wav audio file", type=["wav"])

# Language selection
st.subheader("🌐 Step 2: Choose Language")
lang_option = st.selectbox("Select language spoken in the audio:", {
    "English (India)": "en-IN",
    "Hindi": "hi-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Bengali": "bn-IN"
})

# Transcribe
def transcribe_audio_from_bytes(audio_bytes, language='en-IN'):
    recognizer = sr.Recognizer()
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_bytes)
    with sr.AudioFile("recorded_audio.wav") as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except:
        return "Could not understand audio"

# Parse text (basic Hindi + English)
def parse_text(text, language='en-IN'):
    if language == "hi-IN":
        crop_match = re.search(r"(टमाटर|चावल|गेहूं|प्याज़|आलू)", text)
        price_match = re.search(r"(\d+) रुपये", text)
        quantity_match = re.search(r"मात्रा (\d+)", text)
        market_match = re.search(r"(?:में|के) ([\w\s]+?) (?:की|पर)", text)
        date = datetime.today().date()
        crop = crop_match.group(1) if crop_match else None
    else:
        crop_match = re.search(r"(tomato|wheat|rice|onion|potato)", text.lower())
        market_match = re.search(r"in ([a-zA-Z\s]+?) on", text.lower())
        date_match = re.search(r"on ([a-zA-Z]+\s+\d+)", text.lower())
        price_match = re.search(r"for (\d+) rupees", text.lower())
        quantity_match = re.search(r"quantity (\d+)", text.lower())
        try:
            date = datetime.strptime(date_match.group(1), "%B %d").date()
        except:
            date = datetime.today().date()
        crop = crop_match.group(1).capitalize() if crop_match else None
        market_match = market_match

    return {
        "Crop": crop,
        "Market": market_match.group(1).strip().capitalize() if market_match else None,
        "Date": date,
        "Price": int(price_match.group(1)) if price_match else None,
        "Quantity": int(quantity_match.group(1)) if quantity_match else None,
        "Unit": "Quintal",
        "Currency": "INR"
    }

# Prediction
def predict_price(parsed, dataset_path="crop_dataset.csv"):
    if not os.path.exists(dataset_path):
        return "No dataset found to train model."
    df = pd.read_csv(dataset_path)
    if df.shape[0] < 5:
        return "Need more data to train model."

    df = df.dropna(subset=["Crop", "Market", "Date", "Price", "Quantity"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month

    le_crop = LabelEncoder()
    le_market = LabelEncoder()
    df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
    df["Market_enc"] = le_market.fit_transform(df["Market"])

    X = df[["Crop_enc", "Market_enc", "Day", "Month", "Quantity"]]
    y = df["Price"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    input_crop = le_crop.transform([parsed["Crop"]])[0]
    input_market = le_market.transform([parsed["Market"]])[0]
    input_day = parsed["Date"].day
    input_month = parsed["Date"].month
    input_quantity = parsed["Quantity"]
    X_input = [[input_crop, input_market, input_day, input_month, input_quantity]]
    return round(model.predict(X_input)[0], 2)

# Run the app
if audio_bytes or uploaded_file:
    st.subheader("🧠 Step 3: Transcribe and Analyze")
    if uploaded_file:
        audio_bytes = uploaded_file.read()

    with st.spinner("Transcribing..."):
        transcribed_text = transcribe_audio_from_bytes(audio_bytes, language=lang_option)
        st.success("✅ Transcription Done!")
        st.write("📝 Transcribed Text:")
        st.info(transcribed_text)

        parsed_data = parse_text(transcribed_text, language=lang_option)
        df = pd.DataFrame([parsed_data])
        st.write("📊 Extracted Data:")
        st.dataframe(df)

        if st.button("💾 Save to Dataset"):
            file_exists = os.path.exists("crop_dataset.csv")
            df.to_csv("crop_dataset.csv", mode='a', header=not file_exists, index=False)
            st.success("✅ Saved to crop_dataset.csv")

        if st.button("📈 Predict Crop Price"):
            result = predict_price(parsed_data)
            st.success(f"📊 Predicted Price: ₹{result} per quintal")
