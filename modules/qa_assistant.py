import streamlit as st
import os
from optimum.intel.openvino import OVModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch

@st.cache_resource(show_spinner="🔍 Loading QA model and sentence embeddings...")
def load_models():
    model_dir = "models/qa"
    if not os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        model = OVModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2", export=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = OVModelForQuestionAnswering.from_pretrained(model_dir)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

def chunk_text(text, max_tokens=150):
    sentences = text.split('. ')
    chunks, current_chunk = [], ''
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) < max_tokens:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    chunks.append(current_chunk.strip())
    return chunks

def find_best_context(chunks, question):
    context_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, context_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    return chunks[best_idx]

def answer_question(text, question):
    chunks = chunk_text(text)
    best_context = find_best_context(chunks, question)
    result = qa_pipeline(question=question, context=best_context)
    return result['answer'], best_context
