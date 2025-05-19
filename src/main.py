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
    chunks = split_text (text)

    # 3. Criar embeddings
    vector_store = create_embeddings (chunks)

    # 4. RAG QA Chain
    qa_chain = run_qa_chain (vector_store)

    # 5. Perguntas de exemplo
    questions_empresa = {
        "qual nome e o número de CNPJ de cada empresa do texto? Responda sem inserir informações redundantes, seja específico e coloque apenas o nome encontrado separado por vírgula do cnpj, para casos de mais de um nome, separe por ponto e vírgula"
    }

    for q in questions_empresa:
        print(f"\nPergunta: {q}")
        result = qa_chain({"query": q})
        resposta = result["result"]
        fontes = " || ".join([doc.page_content.replace("\n", " ") for doc in result["source_documents"]])

        print("RESPOSTA:", resposta)
        #print("Fontes:", fontes)

        # Gravar cada empresa e CNPJ separadamente no CSV
        with open (output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer (csvfile)
            registros = [r.strip () for r in resposta.split (";") if r.strip ()]  # separa por ponto e vírgula

            for registro in registros:
                if "," in registro:
                    nome, cnpj = [x.strip () for x in registro.split (",", 1)]  # separa nome e CNPJ
                    writer.writerow ([filename, "Empresa", nome, cnpj])
                else:
                    # Caso o modelo não retorne nome e CNPJ corretamente
                    writer.writerow ([filename, "Empresa", registro, ""])

     # 6. Perguntas sobre as pessoas
    questions_pessoas = {
    "qual nome e o número de CPF de cada pessoa do texto? Responda sem inserir informações redundantes, seja específico e coloque apenas o nome encontrado separado por vírgula do CPF, para casos de mais de um nome, separe por ponto e vírgula"
    }

    for q in questions_pessoas:
        print (f"\nPergunta: {q}")
        result = qa_chain ({"query": q})
        resposta = result["result"]
        fontes = " || ".join ([doc.page_content.replace ("\n", " ") for doc in result["source_documents"]])

        print ("RESPOSTA:", resposta)
        # print("Fontes:", fontes)

        # Gravar cada nome de pessoa e CPF separadamente no CSV
        with open (output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer (csvfile)
            registros = [r.strip () for r in resposta.split (";") if r.strip ()]  # separa por ponto e vírgula

            for registro in registros:
                if "," in registro:
                    nome, cpf = [x.strip () for x in registro.split (",", 1)]  # separa nome e CNPJ
                    writer.writerow ([filename, "Pessoa", nome, cpf])
                else:
                    # Caso o modelo não retorne nome e CNPJ corretamente
                    writer.writerow ([filename, "Pessoa", registro, ""])

    #  Limpar o vector store após o uso
    # 5. Liberar memória e apagar vetor store (em memória ou persistido)
    delete_vector_store(vector_store)

print(" Execução finalizada ")



