# 📚 AI-Powered Interactive Learning Assistant

An all-in-one interactive learning dashboard built using **Python**, **Streamlit**, and **Hugging Face NLP models**, optimized with **Intel OpenVINO** for faster performance. This tool allows students and educators to interact with textbook content using AI-powered summarization, Q&A, test generation, flashcards, and even diagram analysis.

---

## 🚀 Features

### 📁 1. Document Upload
- Uploads PDF, DOCX, or TXT files.
- Extracts and processes the full content for use across all modules.

### 🧠 2. AI Summary
- Uses transformer models (e.g., `t5-small`, `DistilBART`) to summarize lengthy documents into short, coherent explanations.
- Ideal for quick revision and topic overview.

### 🤖 3. Q&A Assistant (Text + Speech Input)
- Ask any question related to the uploaded document.
- Two input modes:
  - **Text input**: Type your question directly.
  - **Speech-to-Text**: Use your mic to ask the question.
- Returns AI-generated answers **based only on the document content**.

### 📝 4. Test Generator & Evaluator
- Automatically generates:
  - **5 Multiple Choice Questions (MCQs)**
  - **5 Fill-in-the-Blank questions**
- Avoids common or irrelevant keywords using a keyword extractor and stopword filtering.
- Accepts your answers and **evaluates your score** out of 10 instantly.

### 📚 5. Question Bank Generator
- Automatically creates a categorized question bank with:
  - 1-mark
  - 2-mark
  - 3-mark
  - 5-mark questions
- Great for exams and practice.

### 🧠 6. Interactive Flashcards
- Extracts important terms from your document and generates:
  - Definitions
  - Meanings
- Presented in collapsible, flashcard-style format to support active recall.

### 🖼️ 7. Diagram/Image Analyzer
- Upload any diagram or chart image (PNG, JPG, JPEG).
- Uses OCR to extract text from the image and provides explanations based on both the **image content** and **uploaded document context**.
- Built using `Tesseract OCR` and Hugging Face language models.

---

## 🧰 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **NLP Models**: Hugging Face Transformers (T5, DistilBART, etc.)
- **Optimization**: [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- **Speech Recognition**: Google Speech Recognition API
- **OCR**: pytesseract (Tesseract OCR engine)
- **PDF/DOCX parsing**: `pdfplumber`, `python-docx`
- **Image Handling**: PIL (Pillow)

---

## 💾 Installation

### ✅ Clone this repository
```bash
git clone https://github.com/your-username/ai-learning-assistant.git
cd ai-learning-assistant
