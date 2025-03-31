import os
from dotenv import load_dotenv
from src.loader import load_pdf_text
from src.splitter import split_text
from src.embedder import create_embeddings
from src.rag_chain import run_qa_chain

# Carrega variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não definida. Adicione ao arquivo .env")

# Caminho do PDF
pdf_path = "input/gold/ID=DOC202502180000A-Escopo.txt"

# 1. Carregar texto do PDF
combined_text = load_pdf_text(pdf_path)

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
