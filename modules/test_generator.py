import streamlit as st
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
kw_model = KeyBERT()

def extract_keywords(text, num_keywords=20):
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=num_keywords
    )
    return [
        kw[0] for kw in keywords
        if kw[0].lower() not in STOPWORDS and len(kw[0].split()) <= 2
    ]

def generate_mcqs(text, count=5):
    sentences = sent_tokenize(text)
    keywords = extract_keywords(text, num_keywords=count * 3)
    questions = []
    used_keywords = set()

    for keyword in keywords:
        if keyword.lower() in used_keywords:
            continue

        for sentence in sentences:
            if keyword in sentence and len(sentence.split()) > 5:
                question_text = sentence.replace(keyword, "____")
                if "____" not in question_text:
                    continue

                distractors = [
                    kw for kw in keywords 
                    if kw != keyword and kw.lower() not in keyword.lower()
                ]
                distractors = list(set(distractors))
                distractors = random.sample(distractors, min(3, len(distractors)))

                options = distractors + [keyword]
                random.shuffle(options)

                questions.append({
                    "question": question_text.strip(),
                    "options": options,
                    "answer": keyword
                })
                used_keywords.add(keyword.lower())
                break

        if len(questions) >= count:
            break

    return questions

def generate_fill_in_the_blanks(text, count=5):
    sentences = sent_tokenize(text)
    keywords = extract_keywords(text, num_keywords=count * 3)
    blanks = []
    used_keywords = set()

    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence and keyword.lower() not in used_keywords:
                blanked = sentence.replace(keyword, "____")
                if "____" in blanked:
                    blanks.append({
                        "question": blanked.strip(),
                        "answer": keyword
                    })
                    used_keywords.add(keyword.lower())
                    break
        if len(blanks) >= count:
            break

    return blanks
