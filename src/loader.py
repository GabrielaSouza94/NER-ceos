import os
from pypdf import PdfReader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

def load_documents_from_input(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f" Pasta não encontrada: {folder_path}")
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith (".txt"):
            print (f" Lendo TXT: {file}")
            try:
                with open (path, "r", encoding="utf-8") as f:
                    documents.append (f.read ())
            except UnicodeDecodeError:
                with open (path, "r", encoding="iso-8859-1") as f:
                    documents.append (f.read ())

        elif file.endswith(".pdf"):
            print(f" Lendo PDF: {file}")
            reader = PdfReader(path)
            texts = []
            for i in range(len(reader.pages)):
                text = reader.pages[i].extract_text()
                if text and text.strip():
                    texts.append(text.strip())
            combined = " ".join(texts)
            documents.append(combined)

        else:
            print(f" Tipo de arquivo não suportado: {file}")

    all_text = "\n".join(documents)
    print(" Texto combinado (início):")
    print(all_text[:500])
    return all_text
