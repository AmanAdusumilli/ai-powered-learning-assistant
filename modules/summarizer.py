import os
import streamlit as st
from transformers import pipeline
from optimum.intel.openvino import OVModelForSeq2SeqLM
from transformers import AutoTokenizer

@st.cache_resource(show_spinner="🧠 Loading summarization model...")
def load_summarizer():
    model_dir = "models/summarizer"
    if not os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = OVModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6", export=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForSeq2SeqLM.from_pretrained(model_dir)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

def generate_summary(text):
    max_chunk = 800
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    results = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(results)
