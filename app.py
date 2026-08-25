import streamlit as st
import imageio_ffmpeg
import os as _os
_os.environ["PATH"] += _os.pathsep + _os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
import tempfile
import re
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
from transformers import pipeline
from langdetect import detect
from textblob import TextBlob


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()


@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_model = load_sentiment_model()

def normalize_lang_code(code):
    code = code.lower()
    mapping = {
        "zh-cn": "zh-CN",
        "zh-tw": "zh-CN",
        "zh": "zh-CN",
    }
    return mapping.get(code, code)

def correct_typos(text, lang_code):
    if lang_code == "en":
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text
    return text

def get_modifiers(text):
    modifiers = []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    question_words = {"what", "why", "how", "when", "where", "who", "which", "whom"}

    has_question = False
    has_exclamation = "!" in text

    for s in sentences:
        s_clean = s.strip().lower()
        if not s_clean:
            continue
        if s_clean.endswith("?"):
            has_question = True
        first_word = s_clean.split()[0].strip(",.!?") if s_clean.split() else ""
        if first_word in question_words:
            has_question = True

    if has_question:
        modifiers.append("Question")
    if has_exclamation:
        modifiers.append("Exclamation")

    return modifiers

def map_sentiment(label, score):
    star = int(label[0])
    if score < 0.35:
        return "Normal"
    mapping = {
        1: "Very Bad",
        2: "Bad",
        3: "Normal",
        4: "Good",
        5: "Very Good"
    }
    return mapping[star]

# --- Session State ---
for key in ["audio_file_path", "transcript", "translated_text", "audio_output", "sentiment",
            "detected_lang_name", "detected_lang_code", "typed_text"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["transcript", "translated_text"] else ""


languages = {
    "Hindi": "hi", "Bengali": "bn", "Marathi": "mr", "Telugu": "te",
    "Tamil": "ta", "Gujarati": "gu", "Urdu": "ur", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Nepali": "ne", "English": "en",
    "Spanish": "es", "Chinese": "zh-CN", "Tagalog": "tl", "Vietnamese": "vi"
}

st.sidebar.header("Translation Settings")
st.sidebar.info("Source language will be auto-detected")
target_lang = st.sidebar.selectbox("Target Language", list(languages.keys()), index=1)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    input_tab1, input_tab2, input_tab3 = st.tabs(["Record", "Upload", "Type Text"])

    with input_tab1:
        from streamlit_mic_recorder import mic_recorder
        audio_record = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key='recorder'
        )
        if audio_record:
            st.audio(audio_record['bytes'])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_record['bytes'])
                st.session_state.audio_file_path = f.name
                st.session_state.input_mode = "audio"

    with input_tab2:
        uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(uploaded_file.getbuffer())
                st.session_state.audio_file_path = f.name
                st.session_state.input_mode = "audio"
            st.audio(uploaded_file)

    with input_tab3:
        typed_text = st.text_area("Type your sentence here", key="typed_input", height=100)
        if typed_text:
            st.session_state.typed_text = typed_text
            st.session_state.input_mode = "text"

    if st.button("Translate"):
        try:
            mode = st.session_state.get("input_mode")

            if mode == "audio" and st.session_state.audio_file_path:
                with st.spinner("Detecting language and transcribing..."):
                    audio_data = whisper.audio.load_audio(st.session_state.audio_file_path)
                    result = model.transcribe(audio_data)
                    detected_code = normalize_lang_code(result["language"])
                    st.session_state.transcript = correct_typos(result["text"], detected_code)
                    reverse_lookup = {v: k for k, v in languages.items()}
                    st.session_state.detected_lang_name = reverse_lookup.get(detected_code, detected_code)
                    st.session_state.detected_lang_code = detected_code

            elif mode == "text" and st.session_state.get("typed_text"):
                try:
                    detected_code = normalize_lang_code(detect(st.session_state.typed_text))
                except Exception:
                    detected_code = "en"
                st.session_state.transcript = correct_typos(st.session_state.typed_text, detected_code)
                reverse_lookup = {v: k for k, v in languages.items()}
                st.session_state.detected_lang_name = reverse_lookup.get(detected_code, detected_code)
                st.session_state.detected_lang_code = detected_code

            else:
                st.warning("Please record or upload audio, or type a sentence first.")
                st.stop()

            src_code = st.session_state.get("detected_lang_code") or "auto"

            with st.spinner(f"Translating to {target_lang}..."):
                translated = GoogleTranslator(
                    source=src_code,
                    target=languages[target_lang]
                ).translate(st.session_state.transcript)
                st.session_state.translated_text = translated

            with st.spinner("Analyzing sentiment..."):
                english_text = GoogleTranslator(
                    source=src_code,
                    target="en"
                ).translate(st.session_state.transcript)

                sentiment_result = sentiment_model(english_text)[0]
                base_sentiment = map_sentiment(sentiment_result['label'], sentiment_result['score'])
                modifiers = get_modifiers(english_text)

                if modifiers:
                    st.session_state.sentiment = f"{base_sentiment}, {', '.join(modifiers)}"
                else:
                    st.session_state.sentiment = base_sentiment

            with st.spinner("Generating audio..."):
                tts = gTTS(text=st.session_state.translated_text, lang=languages[target_lang])
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    with open(f.name, "rb") as audio_f:
                        st.session_state.audio_output = audio_f.read()

        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Output")
    if st.session_state.transcript:
        display_lang = st.session_state.get("detected_lang_name") or "Detected"
        st.write(f"**Original ({display_lang})**")
        st.info(st.session_state.transcript)

    if st.session_state.translated_text:
        st.write(f"**Translated ({target_lang})**")
        st.success(st.session_state.translated_text)

    if st.session_state.sentiment:
        st.write("**Sentiment**")
        st.info(st.session_state.sentiment)

    if st.session_state.audio_output:
        st.audio(st.session_state.audio_output, format="audio/mp3")
        st.download_button("Download", st.session_state.audio_output, "output.mp3", "audio/mp3")

st.caption("SwarSense — Multilingual Speech Translation and Sentiment Analysis (Whisper, Google Translate, gTTS, BERT)")