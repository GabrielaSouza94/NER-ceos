# V1 — dataset Gold legado (sem Pydantic; validação val3 + CSV)

import os
import csv
import json
import datetime
import time
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from loader import load_single_document
from splitter import split_text
from embedder import create_embeddings
from utils_ollama import delete_vector_store, sanitize_text
from parse_resposta_v1 import (
    correction_for_error,
    parse_empresas_response,
    parse_pessoas_response,
)
from val3 import ValidadorEntidadesID

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def invoke_with_retry(chain, prompt, max_retries=8, base_wait=20):
    for attempt in range(max_retries):
        try:
            return chain.invoke(prompt)
        except Exception as e:
            err = str(e).lower()
            retriable = any(k in err for k in [
                "404", "connection", "timeout", "503", "busy",
                "server error", "502", "504", "429", "rate",
                "expecting value", "json",
            ])
            if attempt < max_retries - 1 and retriable:
                wait = min(base_wait * (2 ** attempt), 300)
                print(f"      ⚠ Tentativa {attempt+1}/{max_retries} falhou, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise


start_time = time.time()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

CONFIG_PARAMS = {
    "MODELO": "llama3.3:70b",
    "CHUNK_SIZE": 150,
    "CHUNK_OVERLAP": 50,
    "SEPARATORS": [";"],
    "EMBEDDING_MODEL": "nomic-embed-text",
    "K": 30,
    "TEMPERATURE": 0.1,
    "NUM_PREDICT": 300,
    "PAGES": 4,
    "AUX_TERMS": "sim",
    "PROMPT_ENHANCEMENT": "não",
    "NEGATIVE_PROMPT": "não",
}

_config_json = os.environ.get("CONFIG_JSON")
if _config_json:
    CONFIG_PARAMS.update(json.loads(_config_json))

PROVIDER = os.environ.get("PROVIDER", "ollama")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODELS = {
    "llama3.3:70b": "meta-llama/llama-3.3-70b-instruct",
    "deepseek-r1:70b": "deepseek/deepseek-r1-distill-llama-70b",
    "gemma3:27b": "google/gemma-3-27b-it",
    "qwen3:32b": "qwen/qwen3-32b",
}

PARSE_MAX_RETRIES = int(os.environ.get(
    "PARSE_MAX_RETRIES",
    "5" if CONFIG_PARAMS.get("MODELO") == "gemma3:27b" else "3",
))

PUBLIC_ENTITY_KEYWORDS = [
    "prefeitura", "municipio de", "município de", "camara municipal",
    "câmara municipal", "governo do estado", "governo do municipio",
    "governo federal", "estado de ", "estado do ", "fundacao municipal",
    "fundação municipal", "secretaria municipal", "secretaria de ",
    "tribunal de contas", "ministerio", "ministério", "camara dos deputados",
    "câmara dos deputados", "senado federal", "autarquia",
    "universidade federal", "universidade estadual",
]

timestamp_for_files = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "..", "data", "synthetic_dataset_single_company")

input_folder = os.environ.get("INPUT_FOLDER", _DATA_DIR)
INPUT_FOLDER = input_folder
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_BASE_DIR, "..", "output", "methodology_I"),
)

arquivo_gold = os.environ.get(
    "GOLD_CSV_FILE",
    os.path.join(_DATA_DIR, "gold_single_company.csv"),
)
arquivo_respostas = os.path.join(
    OUTPUT_DIR,
    "V1(2EMP)_extracao_{}_{}_filtrado.csv".format(
        CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files
    ),
)
REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "V1(2EMP)_relatório_{}_{}_filtrado.txt".format(
        CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files
    ),
)
output_csv = arquivo_respostas
REPORT_FILE_JSON = os.path.join(
    OUTPUT_DIR,
    "V1(2EMP)_relatório_{}_{}_filtrado.json".format(
        CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files
    ),
)
CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    os.environ.get(
        "CHECKPOINT_NAME",
        f"V1_checkpoint_{CONFIG_PARAMS['MODELO'].replace(':', '_')}.json",
    ),
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_checkpoint_v1():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = data.get("processed_files", [])
        print(f"  Checkpoint carregado: {len(done)} documento(s) já processado(s)")
        return set(done)
    return set()


def save_checkpoint_v1(processed_files):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_files": sorted(processed_files)}, f, ensure_ascii=False, indent=2)
    print(f"  [Checkpoint salvo: {len(processed_files)} documento(s)]")


def create_llm(temperature, num_predict):
    is_gemma = CONFIG_PARAMS.get("MODELO") == "gemma3:27b"
    if PROVIDER == "openrouter":
        model_id = OPENROUTER_MODELS.get(CONFIG_PARAMS["MODELO"], CONFIG_PARAMS["MODELO"])
        kwargs = {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": 4096,
            "api_key": OPENROUTER_API_KEY,
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": 300,
        }
        if not is_gemma:
            kwargs["model_kwargs"] = {
                "extra_body": {"plugins": [{"id": "context-compression"}]},
            }
        return ChatOpenAI(**kwargs)
    return ChatOllama(
        model=CONFIG_PARAMS["MODELO"],
        base_url=OLLAMA_HOST,
        temperature=temperature,
        num_predict=num_predict,
        timeout=300,
        client_kwargs={"timeout": 300.0},
    )


def run_qa_chain(vector_store, k, temperature, num_predict):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = create_llm(temperature, num_predict)

    prompt = ChatPromptTemplate.from_template(
        """Use as seguintes informações de contexto para responder à pergunta.
Se você não souber a resposta, não responda nada, não tente inventar uma resposta.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""
    )

    def ensure_query(x):
        if isinstance(x, dict):
            return x.get("query") or x.get("question") or ""
        return str(x)

    def combine_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {
            "question": RunnableLambda(ensure_query),
            "context": RunnableLambda(ensure_query) | retriever | RunnableLambda(combine_docs),
        }
        | prompt
        | llm
        | RunnableLambda(lambda msg: {"result": getattr(msg, "content", str(msg))})
    )


def _config_flag(key: str) -> bool:
    return str(CONFIG_PARAMS.get(key, "não")).strip().lower() == "sim"


def build_question_empresas(correction: str | None = None) -> str:
    parts = [
        """Você é um assistente encarregado de extrair informações de documentos investigativos.
Dado o seguinte texto, identifique e extraia:
- Nome completo das empresas que aparecem no texto
- CNPJ das empresas que aparecem no texto

Responda EXCLUSIVAMENTE no formato: NOME1, CNPJ1; NOME2, CNPJ2;
NÃO use JSON. NÃO use null. Se não souber o CNPJ, deixe vazio após a vírgula.""",
    ]
    if _config_flag("AUX_TERMS"):
        parts.append("""Para auxiliar na identificação das empresas procure pelos termos:
- Pessoa jurídica de direito privado
- Sociedade empresária limitada / Sociedade Limitada / LTDA
- Microempresa / ME / EIRELI / S.A. / Fundação / EPP""")
    if _config_flag("NEGATIVE_PROMPT"):
        parts.append("ATENÇÃO: Não invente dados. Não inclua órgãos públicos como empresas.")
    if correction:
        parts.append(f"CORREÇÃO: {correction}")
    return "\n".join(parts)


def build_question_pessoas(correction: str | None = None) -> str:
    parts = [
        """Você é um assistente encarregado de extrair informações de documentos investigativos.
Dado o seguinte texto, identifique e extraia:
- Nomes completo dos sócios e proprietários das empresas que aparecem no texto
- RGs dos sócios e proprietários
- CPFs dos sócios e proprietários

Responda EXCLUSIVAMENTE no formato: NOME1, CPF1, RG1; NOME2, CPF2, RG2;
NÃO use JSON. NÃO use null. Se não souber CPF ou RG, deixe vazio.""",
    ]
    if _config_flag("AUX_TERMS"):
        parts.append("""Para auxiliar na identificação dos sócios procure pelos termos:
- Sócio-administrador / Sócia da empresa
- Representada pelos sócios administradores / De propriedade de
- Gerente / Diretor / Administrador da sociedade""")
    if _config_flag("NEGATIVE_PROMPT"):
        parts.append("ATENÇÃO: Não invente dados que não aparecem no texto.")
    if correction:
        parts.append(f"CORREÇÃO: {correction}")
    return "\n".join(parts)


def _is_public_entity(nome: str) -> bool:
    nome_lower = sanitize_text(nome).lower()
    return any(term in nome_lower for term in PUBLIC_ENTITY_KEYWORDS)


def _write_empresas_csv(writer, filename, rows):
    n = 0
    for nome, cnpj in rows:
        nome_sanitizado = sanitize_text(nome)
        if not nome_sanitizado.strip():
            continue
        if _is_public_entity(nome_sanitizado):
            continue
        writer.writerow([filename, "Empresa", nome_sanitizado, cnpj or "", ""])
        n += 1
    return n


def _write_pessoas_csv(writer, filename, rows):
    n = 0
    for nome, cpf, rg in rows:
        nome_sanitizado = sanitize_text(nome)
        if not nome_sanitizado.strip():
            continue
        writer.writerow([filename, "Pessoa", nome_sanitizado, cpf or "", rg or ""])
        n += 1
    return n


def _extract_with_retry(qa_chain, question_builder, parse_fn, kind: str):
    """Invoca qa_chain com retry de parse (schema_echo, JSON inválido, resposta vazia)."""
    correction = None
    last_error = None

    for attempt in range(PARSE_MAX_RETRIES):
        question = question_builder(correction)
        result = invoke_with_retry(qa_chain, question)
        raw = result.get("result", "") or ""
        rows, err = parse_fn(raw)
        last_error = err

        if rows:
            if attempt > 0:
                print(f"      ✓ Parse OK na tentativa {attempt + 1}")
            return rows, None

        correction = correction_for_error(err, kind)
        if correction and attempt < PARSE_MAX_RETRIES - 1:
            print(f"      ⚠ Parse [{err}] — retry {attempt + 2}/{PARSE_MAX_RETRIES}")
            time.sleep(5)

    return [], last_error


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("EXTRATOR V1 — formato texto CSV (sem Pydantic)")
print(f"PROVIDER: {PROVIDER.upper()} | MODELO: {CONFIG_PARAMS['MODELO']}")
print(f"Checkpoint: {CHECKPOINT_FILE}")
print("=" * 60)

already_done = load_checkpoint_v1()

if already_done:
    print(f"\n>>> RETOMANDO de checkpoint: {len(already_done)} documento(s) já processado(s)")
    for fn in sorted(already_done):
        print(f"    - {fn} (já processado)")
    if not os.path.exists(output_csv):
        with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["arquivo", "tipo_entidade", "nome", "identificador", "aux"])
else:
    with open(output_csv, mode="w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["arquivo", "tipo_entidade", "nome", "identificador", "aux"])

all_files = sorted([f for f in os.listdir(input_folder) if f.endswith((".pdf", ".txt"))])
remaining = [f for f in all_files if f not in already_done]
print(f"\nArquivos restantes: {len(remaining)} de {len(all_files)}")

for i_doc, filename in enumerate(remaining):
    file_path = os.path.join(input_folder, filename)
    print(f"\n=== Processando: {filename} ===")

    try:
        text = load_single_document(file_path, CONFIG_PARAMS["PAGES"])
        if not text:
            continue

        chunks = split_text(
            text,
            CONFIG_PARAMS["CHUNK_SIZE"],
            CONFIG_PARAMS["CHUNK_OVERLAP"],
            CONFIG_PARAMS["SEPARATORS"],
        )
        vector_store = create_embeddings(
            chunks,
            CONFIG_PARAMS["EMBEDDING_MODEL"],
            base_url=OLLAMA_HOST,
        )
        qa_chain = run_qa_chain(
            vector_store,
            CONFIG_PARAMS["K"],
            CONFIG_PARAMS["TEMPERATURE"],
            CONFIG_PARAMS["NUM_PREDICT"],
        )

        with open(output_csv, mode="a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)

            print("--- Extraindo empresas ---")
            empresas, err_e = _extract_with_retry(
                qa_chain, build_question_empresas, parse_empresas_response, "empresa"
            )
            n_e = _write_empresas_csv(writer, filename, empresas)
            if n_e:
                print(f"  ✓ Empresas extraídas: {n_e}")
            elif err_e:
                print(f"  ⚠ Nenhuma empresa parseada [{err_e}]")

            print("--- Extraindo pessoas ---")
            pessoas, err_p = _extract_with_retry(
                qa_chain, build_question_pessoas, parse_pessoas_response, "pessoa"
            )
            n_p = _write_pessoas_csv(writer, filename, pessoas)
            if n_p:
                print(f"  ✓ Pessoas extraídas: {n_p}")
            elif err_p:
                print(f"  ⚠ Nenhuma pessoa parseada [{err_p}]")

        delete_vector_store(vector_store)
        already_done.add(filename)
        save_checkpoint_v1(already_done)

        if i_doc < len(remaining) - 1:
            print("  Aguardando 10s antes do próximo documento...")
            time.sleep(10)

    except Exception as e:
        print(f"\n  ERRO ao processar {filename}: {e}")
        print("  Salvando checkpoint e aguardando 60s antes de continuar...")
        save_checkpoint_v1(already_done)
        time.sleep(60)
        continue

validador = ValidadorEntidadesID(
    arquivo_gabarito=arquivo_gold,
    arquivo_respostas=arquivo_respostas,
    limiar_similaridade=0.95,
    config_params=CONFIG_PARAMS,
)
validador.executar(REPORT_FILE, REPORT_FILE_JSON)

print("Execução finalizada")
elapsed = time.time() - start_time
print(f"\nTempo total de execução: {elapsed:.2f} segundos ({elapsed/60:.2f} minutos)")
