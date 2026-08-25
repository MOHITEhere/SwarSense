# SwarSense — Multilingual Speech Translation & Sentiment Analysis 🎙️🌐

**SwarSense** is a comprehensive, end-to-end NLP and speech processing application built with Streamlit. It seamlessly integrates speech recognition, language identification, neural machine translation, fine-tuned multilingual sentiment classification, and text-to-speech (TTS) synthesis into a unified interface.

---

## 📌 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack & Models](#-tech-stack--models)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Configuration & Performance](#-configuration--performance)
- [License](#-license)

---

## 🌟 Key Features

* **Multi-Modal Input Options:** 
  * **Live Microphone Input:** Record audio directly via standard Web APIs using `streamlit-mic-recorder`.
  * **File Uploads:** Upload pre-recorded audio files (`.wav` and `.mp3`).
  * **Text Input:** Type or paste raw multilingual text directly.
* **Automatic Speech Recognition (ASR):** Uses **OpenAI Whisper** for high-accuracy speech-to-text transcription across diverse accents and noisy environments.
* **Dynamic Language Identification & Correction:** Automatically identifies source languages, normalizes locale tags (e.g., `zh-CN`), and corrects English typos via `TextBlob`.
* **Neural Machine Translation:** Real-time translation into 16+ global and regional languages powered by **GoogleTranslator**.
* **Multilingual Sentiment & Mood Analysis:**
  * Uses a pre-trained multilingual transformer model (`nlptown/bert-base-multilingual-uncased-sentiment`) to evaluate 5-level sentiment intensity (*Very Bad* to *Very Good*).
  * Incorporates rule-based pattern matchers to tag linguistic intent (e.g., detecting *Questions* or *Exclamations*).
* **Text-to-Speech (TTS) Synthesis:** Synthesizes translated output back into natural audio using **gTTS**, with instant web playback and downloadable `.mp3` exports.

---

## 🏗️ System Architecture

                   ┌─────────────────────────┐
                   │  Input Options          │
                   │  (Record / Upload / Text)
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │  OpenAI Whisper (ASR)   │
                   │  & Language Detection   │
                   └────────────┬────────────┘
                                │
                     [Transcribed Text]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
│ GoogleTranslator      │ │ BERT Multilingual │ │ gTTS Audio Generation │
│ (Target Language)     │ │ Sentiment Model   │ │ (.mp3 output)         │
└───────────┬───────────┘ └─────────┬─────────┘ └───────────┬───────────┘
│                       │                       │
└───────────────────────┼───────────────────────┘
│
▼
┌─────────────────────────┐
│   Streamlit Web UI      │
└─────────────────────────┘


---

## 🛠️ Tech Stack & Models

| Component | Library / Framework | Description |
| :--- | :--- | :--- |
| **Frontend Framework** | `Streamlit` | Interactive Python web application framework |
| **Speech-to-Text** | `OpenAI Whisper (base)` | Deep learning ASR model for speech recognition |
| **Sentiment Analysis** | `transformers` (Hugging Face) | `nlptown/bert-base-multilingual-uncased-sentiment` |
| **Translation** | `deep_translator` | Interface for neural machine translation |
| **Text-to-Speech** | `gTTS` | Google Text-to-Speech audio synthesizer |
| **Audio Processing** | `imageio_ffmpeg`, `streamlit_mic_recorder` | Audio decoding and stream capture |
| **NLP Utilities** | `langdetect`, `TextBlob`, `re` | Language identification and text processing |

---

## 📁 Project Directory Structure

```text
SwarSense/
├── app.py                   # Main Streamlit application logic
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── screenshots/             # Interface and output screenshots
    ├── input.png
    └── output.png


⚙️ Configuration & Performance
Model Caching: The application utilizes @st.cache_resource for loading Whisper (base) and BERT (bert-base-multilingual-uncased-sentiment) to minimize latency on re-run operations.

FFmpeg Dependency: Includes imageio_ffmpeg to resolve system audio decoding paths automatically without requiring explicit environment path installation.
