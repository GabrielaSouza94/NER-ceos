import shutil
import os
import gc
import re

def delete_vector_store(vector_store=None, persist_path=None):
    try:
        if vector_store is not None:
            vector_store.delete_collection ()
            del vector_store

        gc.collect ()

        if persist_path and os.path.exists (persist_path):
            shutil.rmtree (persist_path)
            print (f"[✔] Vector store em '{persist_path}' apagado com sucesso.")
        else:
            print (f"[✔] Vector store removido da memória. Nada a limpar em disco.")
        return True
    except Exception as e:
        print (f"[✖] Erro ao limpar vector store: {e}")
        return False

def sanitize_text(text):
    """Remove ou substitui caracteres especiais que podem causar problemas no CSV"""
    # Substitui vírgulas por espaço seguido de hífen para preservar legibilidade
    sanitized = text.replace(",", " -")
    # Pode adicionar mais substituições conforme necessidade
    return sanitized

def clean_llm_response(response):
    """Remove tags de raciocínio e normaliza espaços (fallback texto legado v1)."""
    cleaned = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        response,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    lines = cleaned.split("\n")
    for line in lines:
        line = line.strip()
        if "," in line and (";" in line or line.endswith(",")):
            return line

    return cleaned