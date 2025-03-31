from pypdf import PdfReader

def load_pdf_text(pdf_path, max_pages=4):
    reader = PdfReader(pdf_path)
    texts = []

    for i in range(min(max_pages, len(reader.pages))):
        text = reader.pages[i].extract_text()
        if text and text.strip():
            texts.append(text.strip())

    combined = " ".join(texts)
    print(" Texto processado (início):")
    print(combined[:500])
    return combined
