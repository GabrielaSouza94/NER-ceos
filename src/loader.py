import os
from pypdf import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

def load_single_document(file_path,max_pages=4):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if file_path.endswith(".txt"):
        print(f"Lendo TXT: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="iso-8859-1") as f:
                return f.read()

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
