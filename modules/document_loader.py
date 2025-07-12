import pdfplumber
import docx
import fitz  # PyMuPDF

def load_text_from_txt(file):
    return file.read().decode("utf-8")

def load_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def load_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def load_text(file):
    if file.name.endswith('.pdf'):
        return load_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return load_text_from_docx(file)
    elif file.name.endswith('.txt'):
        return load_text_from_txt(file)
    else:
        return "Unsupported file format"
