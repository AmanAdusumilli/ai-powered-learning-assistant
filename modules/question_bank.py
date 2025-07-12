import os
import streamlit as st
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

@st.cache_resource(show_spinner="📘 Loading Question Generator...")
def load_qg_model():
    model_dir = "models/qbank"
    if not os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
        model = OVModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl", export=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForSeq2SeqLM.from_pretrained(model_dir)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

qg_pipeline = load_qg_model()

def generate_question_bank(text, limit_per_type=3):
    sentences = sent_tokenize(text)
    short, medium, long, essay = [], [], [], []

    for sentence in sentences:
        if len(short) < limit_per_type and 5 < len(sentence.split()) <= 10:
            q = generate_question(sentence)
            if q: short.append(q)
        elif len(medium) < limit_per_type and 10 < len(sentence.split()) <= 15:
            q = generate_question(sentence)
            if q: medium.append(q)
        elif len(long) < limit_per_type and 15 < len(sentence.split()) <= 25:
            q = generate_question(sentence)
            if q: long.append(q)
        elif len(essay) < limit_per_type and len(sentence.split()) > 25:
            q = generate_question(sentence)
            if q: essay.append(q)

        if all(len(lst) >= limit_per_type for lst in [short, medium, long, essay]):
            break

    return {
        "1 Mark": short,
        "2 Mark": medium,
        "3 Mark": long,
        "5 Mark": essay
    }

def generate_question(sentence):
    prompt = f"generate question: {sentence}"
    try:
        result = qg_pipeline(prompt, max_length=64, do_sample=False)[0]['generated_text']
        return result
    except:
        return None
