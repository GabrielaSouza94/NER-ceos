import os
from dotenv import load_dotenv
from src.loader import load_single_document
from src.splitter import split_text
from src.embedder import create_embeddings
from src.rag_chain import run_qa_chain
from src.utils import delete_vector_store
import gc
import csv

#  Limpar o vector store após o uso
#vector_store.delete_collection()
#del vector_store
gc.collect() # força limpeza de memória
delete_vector_store("embeddings/chroma-openai/")

# Carrega variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não definida. Adicione ao arquivo .env")

output_csv = "respostas_llm.csv"
input_folder="../input_files"
with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["arquivo", "tipo_entidade", "nome", "identificador"])  # cabeçalho

# 1. Carregar cada documento separadamente, para fazer o pipeline de perguntas
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if not (file_path.endswith(".pdf") or file_path.endswith(".txt")):
        continue

    print (f"\n=== Processando: {filename} ===")
    text = load_single_document (file_path)
    if not text:
        continue

    # 2. Dividir texto em chunks
    chunks = split_text(text)

    # 3. Criar embeddings
    vector_store = create_embeddings(chunks)

    # 4. RAG QA Chain
    qa_chain = run_qa_chain(vector_store)

    # 5. Perguntas de exemplo
    questions = {
        "nome_empresa": "Qual é o nome da empresa mencionada no documento? Responda sem inserir informações redundantes, seja específico e coloque apenas o nome encontrado",
        "cnpj": "Qual é o CNPJ da empresa, responda sem inserir informações redundantes, seja específico e coloque apenas o número encontrado"
    }

    for q in questions:
        print(f"\nPergunta: {q}")
        result = qa_chain({"query": q})
        resposta = result["result"]
        fontes = " || ".join([doc.page_content.replace("\n", " ") for doc in result["source_documents"]])

        print("RESPOSTA:", resposta)
        print("Fontes:", fontes)

        # Gravar no CSV
        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, q, resposta, fontes])


    #  Limpar o vector store após o uso
    vector_store.delete_collection ()
    del vector_store
    gc.collect ()  # força limpeza de memória
    delete_vector_store ("embeddings/chroma-openai/")


print(" Execução finalizada ")



