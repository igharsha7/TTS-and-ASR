import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
from TTS.api import TTS
from transformers import pipeline
import os

st.title("AI Applications Case Study. Text-to-Speech and Speech-to-Text")
st.header("Text-to-Speech (TTS)")
tts_text = st.text_area("Enter text to convert to speech:")
#TTS Pre-trained model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=True)
SAMPLE_RATE = 22050
if st.button("Convert Text to Speech"):
    if tts_text.strip():
        wav = tts.tts(tts_text, split_sentences=False)
        if isinstance(wav, list):
            wav = np.array(wav)
        if not isinstance(wav, np.ndarray):
            st.error(f"Expected NumPy array, got {type(wav)}")
        else:
            if wav.ndim == 1: #ensures array is 2d
                wav = wav.reshape(-1, 1)    
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sf.write(tmp_wav.name, wav, SAMPLE_RATE)
                st.audio(tmp_wav.name, format="audio/wav")
    else:
        st.warning("Please enter some text.")

#ASR using Whisper
st.header("Speech-to-Text (ASR)")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    file_ext = os.path.splitext(audio_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_audio:
        audio_file.seek(0)
        tmp_audio.write(audio_file.read())
        tmp_audio_path = tmp_audio.name
    try:
        with st.spinner("Transcribing audio... This may take a moment."):
            asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
            result = asr_pipeline(tmp_audio_path)
            st.subheader("Transcript")
            transcript = result.get("text", "")
            if transcript:
                st.write(transcript)
            else:
                st.warning("No transcript was generated. The audio might be too quiet or unclear.")
    
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        st.info("Try a different audio file or check that the file contains clear speech.")
    finally:
        if os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)