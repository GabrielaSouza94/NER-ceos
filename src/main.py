import os
from dotenv import load_dotenv
from src.loader import load_single_document
from src.splitter import split_text
from src.embedder import create_embeddings
from src.rag_chain import run_qa_chain
from src.utils import delete_vector_store
from src.utils import sanitize_text
from src.validador_estat import validar_entidades_simples
import gc
import csv
import sys
import datetime

# Carrega variáveis de ambiente
load_dotenv ()
openai_api_key = os.getenv ("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError ("OPENAI_API_KEY não definida. Adicione ao arquivo .env")

# parametros ajustáveis
CONFIG_PARAMS = {
    "MODEL": "gpt-3.5-turbo",  # Add the model name
    "EMBEDDING_MODEL": "text-embedding-ada-002",  # Add embedding model
    "CHUNK_SIZE": 800,
    "CHUNK_OVERLAP": 300,
    "SEPARATORS": [";"],
    "K": 10,  # Number of documents retrieved for RAG
    "TEMPERATURE": 0,  # Temperature for the model
    "NUM_PREDICT": 'none'  # Max tokens parameter

}

# definir caminho dos arquivos input e output
output_csv = "../out_files/respostas_llm.csv".format(CONFIG_PARAMS["MODEL"])
input_folder = "../input_files"
arquivo_gold = "../out_files/gold_teste.csv"
arquivo_respostas = output_csv
consolidated_report_file = "../out_files/consolidated_validation_report.txt".format(CONFIG_PARAMS["MODEL"])

# Create output directory if it doesn't exist
os.makedirs ("../out_files", exist_ok=True)

with open (output_csv, mode="w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer (csvfile)
    writer.writerow (["arquivo", "tipo_entidade", "nome", "identificador", "aux"])  # cabeçalho

# 1. Carregar cada documento separadamente, para fazer o pipeline de perguntas
for filename in os.listdir (input_folder):
    file_path = os.path.join (input_folder, filename)
    if not (file_path.endswith (".pdf") or file_path.endswith (".txt")):
        continue

    print (f"\n=== Processando: {filename} ===")
    text = load_single_document (file_path)
    if not text:
        continue

    # 2. Dividir texto em chunks
    chunks = split_text (text, CONFIG_PARAMS['CHUNK_SIZE'], CONFIG_PARAMS['CHUNK_OVERLAP'], CONFIG_PARAMS['SEPARATORS'])

    # 3. Criar embeddings
    vector_store = create_embeddings (chunks)

    # 4. RAG QA Chain
    qa_chain = run_qa_chain (vector_store, CONFIG_PARAMS['K'], CONFIG_PARAMS['TEMPERATURE'])

    # 5. Perguntas de exemplo
    questions_empresa = [
        """Você é um assistente encarregado de extrair informações de documentos investigativos.
        Dado o seguinte texto, identifique e extraia:
        - Nome completo das empresas que aparecem no texto
        - CNPJ das empresas que aparecem no texto

        Responda EXCLUSIVAMENTE no formato: NOME1, CNPJ1; NOME2, CNPJ2;
        Para auxiliar na identificação das empresas procure pelos jargões:
        - pessoa jurídica de direito privado
        - pessoa jurídica de direito público
        - Sociedade Limitada
        - LTDA
        - ME."""
    ]


    def sanitize_text(text):
        """Remove ou substitui caracteres especiais que podem causar problemas no CSV"""
        # Substitui vírgulas por espaço seguido de hífen para preservar legibilidade
        sanitized = text.replace (",", " ")
        # Pode adicionar mais substituições conforme necessidade
        return sanitized


    for q in questions_empresa:
        #print (f"\nPergunta: {q}")
        result = qa_chain ({"query": q})
        resposta = result["result"]
        fontes = " || ".join ([doc.page_content.replace ("\n", " ") for doc in result["source_documents"]])

        #print ("RESPOSTA:", resposta)
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
    questions_pessoas = [
        """Você é um assistente encarregado de extrair informações de documentos investigativos.
            Dado o seguinte texto, identifique e extraia:
            - Nomes completo dos sócios e proprietários das empresas que aparecem no texto
            - RGs dos sócios e proprietários
            - CPFs dos sócios e proprietários

            Responda EXCLUSIVAMENTE no formato: NOME1, CPF1, RG1; NOME2, CPF2, RG2;
            Para auxiliar na identificação dos sócios procure pelos jargões: 
            - Representado por
            - Seu representante legal 
            - por seus sócios-administradores"""
    ]

    for q in questions_pessoas:
        #print (f"\nPergunta: {q}")
        result = qa_chain ({"query": q})
        resposta = result["result"]
        fontes = " || ".join ([doc.page_content.replace ("\n", " ") for doc in result["source_documents"]])

        #print ("RESPOSTA:", resposta)
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

    # Limpar o vector store após o uso
    delete_vector_store (vector_store)


def generate_consolidated_validation_report():
    """
    Generates a consolidated validation report with all statistics and configuration details
    """
    print (f"\n{'=' * 60}")
    print ("GENERATING CONSOLIDATED VALIDATION REPORT")
    print (f"{'=' * 60}")

    # Get current timestamp
    timestamp = datetime.datetime.now ().strftime ("%Y-%m-%d %H:%M:%S")

    # Start building the consolidated report
    report_content = []
    report_content.append ("=" * 80)
    report_content.append ("CONSOLIDATED VALIDATION REPORT")
    report_content.append ("=" * 80)
    report_content.append (f"Generated on: {timestamp}")
    report_content.append ("")

    # Add configuration parameters
    report_content.append ("SYSTEM CONFIGURATION PARAMETERS:")
    report_content.append ("-" * 40)
    report_content.append (f"MODEL: {CONFIG_PARAMS['MODEL']}")
    report_content.append (f"EMBEDDING_MODEL: {CONFIG_PARAMS['EMBEDDING_MODEL']}")
    report_content.append (f"CHUNK_SIZE: {CONFIG_PARAMS['CHUNK_SIZE']}")
    report_content.append (f"CHUNK_OVERLAP: {CONFIG_PARAMS['CHUNK_OVERLAP']}")
    report_content.append (f"SEPARATORS: {CONFIG_PARAMS['SEPARATORS']}")
    report_content.append (f"K (Retrieved Documents): {CONFIG_PARAMS['K']}")
    report_content.append (f"TEMPERATURE: {CONFIG_PARAMS['TEMPERATURE']}")
    report_content.append (f"NUM_PREDICT: {CONFIG_PARAMS['NUM_PREDICT']}")
    report_content.append ("")

    # Run validation
    print ("Running Validation (validar_entidades_simples)...")
    report_content.append ("RESULTADOS VALIDAÇÃO (validar_entidades_simples):")
    report_content.append ("-" * 40)

    try:
        # Run the validation
        estatisticas = validar_entidades_simples (
            arquivo_gold=arquivo_gold,
            arquivo_respostas=arquivo_respostas,
            output_file=None,
            imprimir=False
        )

        if estatisticas:
            # Overall statistics
            report_content.append (f"Total_entidades: {estatisticas['total_entidades']}")
            report_content.append (
                f"Entidades_encontradas: {estatisticas['entidades_encontradas']} ({estatisticas['taxa_acerto']:.2%})")
            report_content.append (f"   - Via nome+id: {estatisticas['encontradas_nome_id']}")
            report_content.append (f"   - Via nome+aux: {estatisticas['encontradas_nome_auxiliar']}")
            report_content.append (f"Não_encontradas: {estatisticas['entidades_nao_encontradas']}")
            report_content.append ("")

            # Statistics by entity type
            report_content.append ("ESTATÍSTICAS POR TIPO DE ENTIDADE:")
            for tipo, stats in estatisticas['por_tipo'].items ():
                taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
                report_content.append (f"  {tipo.upper ()}:")
                report_content.append (f"    Total: {stats['total']}")
                report_content.append (f"    Encontradas: {stats['encontradas']} ({taxa:.2%})")
                report_content.append (f"    Via nome+id: {stats['nome_id']}")
                report_content.append (f"    Via nome+aux: {stats['nome_auxiliar']}")

            report_content.append ("")

            # Statistics by file
            report_content.append ("STATISTICS BY FILE:")
            for arquivo, stats in estatisticas['por_arquivo'].items ():
                taxa = stats['encontradas'] / stats['total'] if stats['total'] > 0 else 0
                report_content.append (f"  {arquivo}:")
                report_content.append (f"    Total entidades: {stats['total']}")
                report_content.append (f"    Entidades encontradas: {stats['encontradas']} ({taxa:.2%})")
                report_content.append (f"    Pessoas: {stats['pessoas']}")
                report_content.append (f"    Empresas: {stats['empresas']}")
                report_content.append (f"    CPFs: {stats['cpfs']}")
                report_content.append (f"    CNPJs: {stats['cnpjs']}")

                # Warnings for files with no entities found
                if stats['pessoas'] == 0 and stats['total'] > 0:
                    report_content.append (f"    WARNING: Nenhuma pessoa identificada no arquivo {arquivo}")
                if stats['empresas'] == 0 and stats['total'] > 0:
                    report_content.append (f"    WARNING: Nenhuma empresa identificada no arquivo {arquivo}")

            report_content.append ("")

            # Summary metrics
            report_content.append ("RESUMO MÉTRICASS:")
            report_content.append (f"Acurácia geral: {estatisticas['taxa_acerto']:.2%}")

            # Calculate file-level statistics
            files_with_perfect_score = sum (1 for stats in estatisticas['por_arquivo'].values ()
                                            if stats['encontradas'] == stats['total'] and stats['total'] > 0)
            total_files = len (estatisticas['por_arquivo'])

            report_content.append (f"Arquivos com máxima acurácia: {files_with_perfect_score}/{total_files}")

            # Average per-file accuracy
            file_accuracies = [stats['encontradas'] / stats['total'] for stats in estatisticas['por_arquivo'].values ()
                               if stats['total'] > 0]
            avg_file_accuracy = sum (file_accuracies) / len (file_accuracies) if file_accuracies else 0
            report_content.append (f"Medida de acurácia por arquivo: {avg_file_accuracy:.2%}")

        else:
            report_content.append ("Error: Falha na validação")

    except Exception as e:
        report_content.append (f"Error: Falha na validação: {str (e)}")

    report_content.append ("")
    report_content.append ("=" * 80)
    report_content.append ("FINALIZADO")
    report_content.append ("=" * 80)

    # Write the consolidated report to file
    with open (consolidated_report_file, 'w', encoding='utf-8') as f:
        f.write ('\n'.join (report_content))

    # Also print a summary to console
    print (f"\n{'=' * 60}")
    print ("CONSOLIDATED VALIDATION SUMMARY")
    print (f"{'=' * 60}")
    for line in report_content:
        print (line)

    print (f"\nFull consolidated report saved to: {consolidated_report_file}")

    return consolidated_report_file


# Run the consolidated validation
generate_consolidated_validation_report ()

print ("Execução finalizada")