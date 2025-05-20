import os
from dotenv import load_dotenv
from src.loader import load_single_document
from src.splitter import split_text
from src.embedder import create_embeddings
from src.rag_chain import run_qa_chain
from src.utils import delete_vector_store
from src.utils import sanitize_text
from src.validator import avaliar_extracao
from validator2 import validar_entidades_simples
import gc
import csv
import sys
import os



# Carrega variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY não definida. Adicione ao arquivo .env")

#definir caminho dos arquivos input e output
output_csv = "../out_files/respostas_llm.csv"
input_folder="../input_files"
with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["arquivo", "tipo_entidade", "nome", "identificador", "aux"])  # cabeçalho

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


    def sanitize_text(text):
        """Remove ou substitui caracteres especiais que podem causar problemas no CSV"""
        # Substitui vírgulas por espaço seguido de hífen para preservar legibilidade
        sanitized = text.replace (",", " ")
        # Pode adicionar mais substituições conforme necessidade
        return sanitized


    for q in questions_empresa:
        print (f"\nPergunta: {q}")
        result = qa_chain ({"query": q})
        resposta = result["result"]
        fontes = " || ".join ([doc.page_content.replace ("\n", " ") for doc in result["source_documents"]])

        print ("RESPOSTA:", resposta)
        # print("Fontes:", fontes)

        # Gravar cada empresa e CNPJ separadamente no CSV
        with open (output_csv, mode="a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer (csvfile)
            registros = [r.strip () for r in resposta.split (";") if r.strip ()]  # separa por ponto e vírgula

            for registro in registros:
                if "," in registro:
                    partes = registro.split (",", 1)  # separa nome e CNPJ
                    if len (partes) == 2:
                        nome, cnpj = [x.strip () for x in partes]
                        # Sanitizar o nome antes de gravar (para evitar vírgulas extras)
                        nome_sanitizado = sanitize_text (nome)
                        writer.writerow ([filename, "Empresa", nome_sanitizado, cnpj])
                    else:
                        writer.writerow ([filename, "Empresa", sanitize_text (registro), ""])
                else:
                    # Caso o modelo não retorne nome e CNPJ corretamente
                    writer.writerow ([filename, "Empresa", sanitize_text (registro), ""])

    # 6. Perguntas sobre as pessoas
    questions_pessoas = {
        "qual nome é o número de CPF e RG de cada pessoa do texto? Responda sem inserir informações redundantes, seja específico e coloque apenas o nome encontrado separado por vírgula do CPF e do RG, para casos de mais de um nome, separe por ponto e vírgula"
    }

    for q in questions_pessoas:
        print (f"\nPergunta: {q}")
        result = qa_chain ({"query": q})
        resposta = result["result"]
        fontes = " || ".join ([doc.page_content.replace ("\n", " ") for doc in result["source_documents"]])

        print ("RESPOSTA:", resposta)
        # print("Fontes:", fontes)

        # Gravar cada nome de pessoa, CPF e RG separadamente no CSV
        with open (output_csv, mode="a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer (csvfile)
            registros = [r.strip () for r in resposta.split (";") if r.strip ()]  # separa por ponto e vírgula

            for registro in registros:
                partes = [x.strip () for x in registro.split (",", 2)]  # tenta separar em 3 partes

                # Inicializar valores padrão
                nome = cpf = rg = ""

                # Preencher com os valores disponíveis
                if len (partes) >= 1:
                    nome = sanitize_text (partes[0])
                if len (partes) >= 2:
                    cpf = partes[1]
                if len (partes) >= 3:
                    rg = partes[2]

                # Sempre gravar exatamente 5 colunas
                writer.writerow ([filename, "Pessoa", nome, cpf, rg])

    #  Limpar o vector store após o uso
    # 5. Liberar memória e apagar vetor store (em memória ou persistido)
    delete_vector_store(vector_store)


print (f"\n === VALIDADOR DE ESTATÍSTICAS 1 ===")
arquivo_gold = "../out_files/gold_teste.csv"
arquivo_respostas = "../out_files/respostas_llm.csv"

stats = avaliar_extracao(arquivo_gold, arquivo_respostas)
print (stats)

print (f"\n === VALIDADOR DE ESTATÍSTICAS 2 ===")
# Opcionalmente, definir um arquivo de saída para os resultados
arquivo_saida = "../out_files/resultados_validacao.txt"

# Chamar o validador
print ("Iniciando validação de entidades...")
estatisticas = validar_entidades_simples (
    arquivo_gold=arquivo_gold,
    arquivo_respostas=arquivo_respostas,
    output_file=arquivo_saida,
    imprimir=True  # Se quiser imprimir estatísticas na tela
)

if estatisticas:
    # Aqui você pode fazer qualquer processamento adicional com as estatísticas
    print (f"Taxa de acerto: {estatisticas['taxa_acerto']:.2%}")

    # Exemplo: Verificar estatísticas específicas por arquivo
    print ("\nResumo de estatísticas por arquivo:")
    for arquivo, stats in estatisticas['por_arquivo'].items ():
        taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
        print (f"{arquivo}: {taxa:.2%} de acerto ({stats['encontradas']}/{stats['total']})")

        # Verificações específicas
        if stats['pessoas'] == 0 and stats['total'] > 0:
            print (f"  ATENÇÃO: Nenhuma pessoa identificada no arquivo {arquivo}")

        if stats['empresas'] == 0 and stats['total'] > 0:
            print (f"  ATENÇÃO: Nenhuma empresa identificada no arquivo {arquivo}")
else:
    print ("Erro ao executar a validação.")


print(" Execução finalizada ")



