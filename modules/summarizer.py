import streamlit as st
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM

def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = OVModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6", export=True)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

def generate_summary(text):
    max_chunk = 800
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    results = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(results)
