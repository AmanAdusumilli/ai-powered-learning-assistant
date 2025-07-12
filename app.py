import streamlit as st
from modules.document_loader import load_text
from modules.summarizer import generate_summary
from modules.qa_assistant import answer_question
from modules.test_generator import generate_mcqs, generate_fill_in_the_blanks
from modules.question_bank import generate_question_bank
from modules.flashcards import generate_flashcards
from modules.image_analyzer import analyze_image
from PIL import Image
import speech_recognition as sr

# Configure page
st.set_page_config(page_title="AI Learning Assistant", layout="wide")
st.title("📚 AI-Powered Learning Assistant")

# Initialize session state
for key in ["summary", "question", "answer", "context_used", "mcqs", "blanks", "test_submitted",
            "mcq_answers", "blank_answers", "question_bank", "flashcards", "image_explanation"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["mcq_answers", "blank_answers"] else {}

# Upload section
st.markdown("---")
st.markdown("### 📁 Upload Document")
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
text = ""
if uploaded_file:
    text = load_text(uploaded_file)
    st.success("✅ Document loaded successfully!")

# ------------------------ Summary ------------------------
with st.expander("🧠 AI Summary", expanded=True):
    if st.button("Generate Summary"):
        if not text.strip():
            st.error("❌ Please upload a document first.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    short_text = text[:3000]
                    st.session_state.summary = generate_summary(short_text)
                    st.success("✅ Summary generated.")
                except Exception as e:
                    st.error(f"⚠️ Summary generation failed: {e}")
    if st.session_state.summary:
        st.markdown("### 📄 Summary:")
        st.write(st.session_state.summary)

# ------------------------ Q&A ------------------------
with st.expander("🤖 Q&A Assistant", expanded=False):
    mode = st.radio("Choose Input Mode", ["Text", "Speech"], horizontal=True)

    if mode == "Text":
        st.session_state.question = st.text_input("Enter your question", value=st.session_state.question or "")
    elif mode == "Speech":
        if st.button("🎙️ Record"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎤 Listening...")
                try:
                    audio = r.listen(source, timeout=5)
                    recognized = r.recognize_google(audio)
                    st.session_state.question = recognized
                    st.success(f"🗣 You asked: **{recognized}**")
                except sr.WaitTimeoutError:
                    st.error("⏱️ No speech detected.")
                except sr.UnknownValueError:
                    st.error("❌ Could not understand.")
                except sr.RequestError:
                    st.error("⚠️ API request failed.")

    if st.session_state.question and st.button("💬 Get Answer"):
        if not text.strip():
            st.error("❌ Please upload a document.")
        else:
            with st.spinner("Answering..."):
                try:
                    st.session_state.answer, st.session_state.context_used = answer_question(
                        text, st.session_state.question
                    )
                    st.success("✅ Answer generated.")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    if st.session_state.answer:
        st.markdown("### 💡 Answer:")
        st.write(st.session_state.answer)
        st.markdown("### 🔍 Context:")
        st.write(st.session_state.context_used)

# ------------------------ Test ------------------------
with st.expander("📝 Test Generator & Evaluator", expanded=False):
    if st.button("Generate Test (5 MCQs & 5 Fill-in-the-Blanks)"):
        if not text.strip():
            st.error("❌ Please upload a document.")
        else:
            with st.spinner("Generating test..."):
                st.session_state.mcqs = generate_mcqs(text)
                st.session_state.blanks = generate_fill_in_the_blanks(text)
                st.session_state.test_submitted = False
                st.session_state.mcq_answers = {}
                st.session_state.blank_answers = {}
                st.success("✅ Test generated.")

    if st.session_state.mcqs and not st.session_state.test_submitted:
        st.markdown("### 🎯 MCQs")
        for i, q in enumerate(st.session_state.mcqs, 1):
            st.markdown(f"**{i}. {q['question']}**")
            st.session_state.mcq_answers[i] = st.radio("Choose one:", q["options"], key=f"mcq_{i}")

        st.markdown("### ✏️ Fill in the Blanks")
        for i, q in enumerate(st.session_state.blanks, 1):
            st.session_state.blank_answers[i] = st.text_input(f"{i}. {q['question']}", key=f"blank_{i}")

        if st.button("✅ Submit Test"):
            st.session_state.test_submitted = True

    if st.session_state.test_submitted:
        score = 0
        st.markdown("## 📊 Results")

        st.markdown("### 🎯 MCQs")
        for i, q in enumerate(st.session_state.mcqs, 1):
            selected = st.session_state.mcq_answers[i]
            correct = q['answer']
            if selected == correct:
                st.success(f"{i}. ✅ Correct!")
                score += 1
            else:
                st.error(f"{i}. ❌ Incorrect. Your Answer: {selected} | Correct: {correct}")

        st.markdown("### ✏️ Fill in the Blanks")
        for i, q in enumerate(st.session_state.blanks, 1):
            typed = st.session_state.blank_answers[i].strip().lower()
            correct = q['answer'].strip().lower()
            if typed == correct:
                st.success(f"{i}. ✅ Correct!")
                score += 1
            else:
                st.error(f"{i}. ❌ Incorrect. Your Answer: {typed} | Correct: {correct}")

        st.markdown(f"### 🏁 Final Score: **{score}/10**")

# ------------------------ Question Bank ------------------------
with st.expander("📚 Question Bank Generator", expanded=False):
    if st.button("Generate Question Bank"):
        if not text.strip():
            st.error("❌ Please upload a document.")
        else:
            with st.spinner("Generating questions..."):
                st.session_state.question_bank = generate_question_bank(text)
                st.success("✅ Question bank generated.")

    if st.session_state.question_bank:
        for mark, questions in st.session_state.question_bank.items():
            st.markdown(f"### {mark} Questions:")
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")

# ------------------------ Flashcards ------------------------
with st.expander("🧠 Interactive Flashcards", expanded=False):
    if st.button("Generate Flashcards"):
        if not text.strip():
            st.error("❌ Please upload a document.")
        else:
            with st.spinner("Creating flashcards..."):
                st.session_state.flashcards = generate_flashcards(text)
                if not st.session_state.flashcards:
                    st.warning("⚠️ No flashcards generated.")
                else:
                    st.success(f"✅ {len(st.session_state.flashcards)} flashcards created.")

    if st.session_state.flashcards:
        for i, card in enumerate(st.session_state.flashcards, 1):
            with st.expander(f"🃏 {i}. {card['term']}"):
                st.markdown(f"**Definition:** {card['definition']}")

# ------------------------ Image Analyzer ------------------------
with st.expander("🖼️ Diagram/Image Analyzer", expanded=False):
    uploaded_image = st.file_uploader("Upload a diagram or chart", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                st.session_state.image_explanation = analyze_image(image, text)
                st.success("✅ Image analyzed.")

    if st.session_state.image_explanation:
        st.markdown("### 📋 Explanation:")
        st.write(st.session_state.image_explanation)
