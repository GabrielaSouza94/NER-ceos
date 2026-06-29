import os
from pypdf import PdfReader

def load_single_document(file_path, max_pages, max_txt_chars=50000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if file_path.endswith(".txt"):
        print(f"Lendo TXT: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="iso-8859-1") as f:
                text = f.read()
        if max_txt_chars and len(text) > max_txt_chars:
            print(f"  ⚠ TXT truncado: {len(text)} → {max_txt_chars} caracteres")
            text = text[:max_txt_chars]
        return text

    elif file_path.endswith(".pdf"):
        print(f"Lendo PDF: {file_path}")
        reader = PdfReader(file_path)
        texts = []
        for i in range(min(max_pages, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text and text.strip():
                texts.append(text.strip())
        return " ".join(texts)

    else:
        print(f"Tipo de arquivo não suportado: {file_path}")
        return None
