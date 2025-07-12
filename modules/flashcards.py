import os
import streamlit as st
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM

# Initialize KeyBERT model
kw_model = KeyBERT()

@st.cache_resource(show_spinner="🧠 Loading flashcard model...")
def load_explainer_model():
    model_dir = "models/flashcard_explainer"
    
    if not os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = OVModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", export=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForSeq2SeqLM.from_pretrained(model_dir)
    
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load once and use across app
explainer = load_explainer_model()

def generate_flashcards(text, count=10):
    # Extract 2x keywords to allow for filtering
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=count * 3
    )
    
    flashcards = []
    used = set()

    for kw, _ in keywords:
        clean_kw = kw.strip()
        if clean_kw.lower() in used or len(clean_kw.split()) > 3:
            continue

        prompt = f"What is {clean_kw}? Give a short and clear definition."

        try:
            result = explainer(prompt, max_length=70, do_sample=False)[0]['generated_text'].strip()
            print(f"[DEBUG] Flashcard generated → {clean_kw}: {result}")  # Optional: remove in prod
            
            if len(result) > 10:
                flashcards.append({
                    "term": clean_kw,
                    "definition": result
                })
                used.add(clean_kw.lower())
        except Exception as e:
            print(f"[ERROR] Failed for {clean_kw}: {e}")
            continue

        if len(flashcards) >= count:
            break

    return flashcards
