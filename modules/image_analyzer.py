import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM
import streamlit as st

# Load BLIP image captioning model
@st.cache_resource(show_spinner="🔍 Loading image caption model...")
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load Flan-T5 with OpenVINO
@st.cache_resource(show_spinner="🧠 Loading explanation model...")
def load_explainer_model():
    model_dir = "models/image_explainer"
    if not os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = OVModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", export=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForSeq2SeqLM.from_pretrained(model_dir)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

caption_processor, caption_model = load_caption_model()
explainer = load_explainer_model()

# Main function to analyze image and explain
def analyze_image(image: Image.Image, document_text: str) -> str:
    try:
        # Step 1: Generate caption from image
        inputs = caption_processor(image, return_tensors="pt")
        out = caption_model.generate(**inputs, max_new_tokens=50)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Step 2: Generate explanation using the document context
        prompt = (
            f"Based on the document below, explain the image:\n\n"
            f"Document:\n{document_text[:2000]}\n\n"
            f"Image Caption:\n{caption}"
        )

        explanation = explainer(prompt, max_length=150, do_sample=False)[0]['generated_text']
        return explanation.strip()

    except Exception as e:
        return f"❌ Error analyzing image: {e}"
