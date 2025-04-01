import os
from dotenv import load_dotenv
from src.loader import load_documents_from_input
from src.splitter import split_text
from src.embedder import create_embeddings
from src.rag_chain import run_qa_chain
from src.utils import delete_vector_store
import gc


# Carrega variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não definida. Adicione ao arquivo .env")

# 1. Carregar todos os documentos (PDF e TXT)
combined_text = load_documents_from_input("../input_files/gold")

# 2. Dividir texto em chunks
chunks = split_text(combined_text)

# 3. Criar embeddings
vector_store = create_embeddings(chunks)

# 4. RAG QA Chain
qa_chain = run_qa_chain(vector_store)

# 5. Perguntas de exemplo
questions = ["Qual é o escopo do projeto?"]

for q in questions:
    print(f"\n Pergunta: {q}")
    result = qa_chain({"query": q})
    print(" RESPOSTA:", result["result"])

    print("\n Fontes:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"Trecho {i}: {doc.page_content}")

print(" Execução finalizada ")

#  Limpar o vector store após o uso
vector_store.delete_collection()
del vector_store
gc.collect() # força limpeza de memória
delete_vector_store("embeddings/chroma-openai/")

