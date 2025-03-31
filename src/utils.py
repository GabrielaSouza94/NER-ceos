from PyPDF2 import PdfReader

def read_pdf(path):
    pdf_texts = []
    reader = PdfReader(path)

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pdf_texts.append(text.strip())

    combined = " ".join(pdf_texts)
    print("📄 Texto extraído do PDF (início):")
    print(combined[:500])
    return combined
