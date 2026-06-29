### VERSÃO code v3

import os
import re
import time
import datetime
import json
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

def _patch_openai_model_dump():
    import openai
    _orig = openai.BaseModel.model_dump
    def _model_dump_no_warnings(self, **kwargs):
        kwargs.setdefault("warnings", False)
        return _orig(self, **kwargs)
    openai.BaseModel.model_dump = _model_dump_no_warnings

_patch_openai_model_dump()

from loader import load_single_document
from splitter import split_text
from embedder import create_embeddings
from utils_ollama import delete_vector_store

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from validador_json3 import ValidadorDocumentos
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def invoke_with_retry(chain, prompt, max_retries=5, base_wait=15):
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
                wait = base_wait * (2 ** attempt)
                print(f"      ⚠ Tentativa {attempt+1}/{max_retries} falhou, aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise


def invoke_structured_json(llm, prompt_text, schema_class, max_retries=3):
    """Invoca o LLM pedindo resposta em JSON e valida com Pydantic.

    Substitui llm.with_structured_output() para compatibilidade com modelos
    que não suportam tool-calling (ex: Llama via OpenRouter/vLLM).
    """
    schema_def = json.dumps(schema_class.model_json_schema(), ensure_ascii=False, indent=2)
    full_prompt = (
        f"{prompt_text}\n\n"
        "IMPORTANTE: Responda APENAS com um objeto JSON válido, sem texto antes ou depois.\n"
        f"O JSON deve seguir exatamente esta estrutura:\n{schema_def}"
    )

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = invoke_with_retry(llm, full_prompt)
            content = response.content.strip()

            # Remove blocos <think>...</think> (modelos DeepSeek-R1)
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

            # Extrai JSON de blocos markdown ```json ... ```
            md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
            if md_match:
                content = md_match.group(1).strip()

            # Garante que pegamos apenas o objeto JSON principal
            obj_match = re.search(r"\{[\s\S]*\}", content)
            if obj_match:
                content = obj_match.group(0)

            data = json.loads(content)
            return schema_class(**data)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(5)

    raise last_exc


# ── Configurações ──────────────────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

CONFIG_PARAMS = {
    "MODELO": "llama3.3:70b",
    "CHUNK_SIZE": 150,
    "CHUNK_OVERLAP": 50,
    "SEPARATORS": [";"],
    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "K": 5,
    "TEMPERATURE": 0.1,
    "NUM_PREDICT": 1024,
    "PAGES": 7,
    "MAX_TXT_CHARS": 50000,
    "MAX_CONTEXT_CHARS": 30000,
    "AUX_TERMS": "sim",
    "PROMPT_ENHANCEMENT": "não",
    "NEGATIVE_PROMPT": "não"
}

_config_json = os.environ.get("CONFIG_JSON")
if _config_json:
    CONFIG_PARAMS.update(json.loads(_config_json))

# ── Provider: "ollama" (servidor local) ou "openrouter" (nuvem) ────────────────
# Defina o provider via variável de ambiente PROVIDER (padrão: "openrouter").
# A chave da API deve ser fornecida via variável de ambiente OPENROUTER_API_KEY.
PROVIDER = os.environ.get("PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_MODELS = {
    "llama3.3:70b":     "meta-llama/llama-3.3-70b-instruct",
    "deepseek-r1:70b":  "deepseek/deepseek-r1-distill-llama-70b",
    "gemma3:27b":       "google/gemma-3-27b-it",
    "qwen3:32b":        "qwen/qwen3-32b",
}


def create_llm(temperature=None, num_predict=None):
    """Cria o LLM via OpenRouter (nuvem) ou Ollama (servidor local), conforme PROVIDER."""
    t = temperature if temperature is not None else CONFIG_PARAMS["TEMPERATURE"]
    n = num_predict if num_predict is not None else CONFIG_PARAMS["NUM_PREDICT"]

    if PROVIDER == "openrouter":
        model_id = OPENROUTER_MODELS.get(CONFIG_PARAMS["MODELO"], CONFIG_PARAMS["MODELO"])
        return ChatOpenAI(
            model=model_id,
            temperature=t,
            max_tokens=2048,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=300,
            model_kwargs={
                "extra_body": {
                    "plugins": [{"id": "context-compression"}],
                }
            },
        )
    else:
        return ChatOllama(
            model=CONFIG_PARAMS["MODELO"],
            base_url=OLLAMA_HOST,
            temperature=t,
            num_predict=n,
            timeout=300,
            client_kwargs={"timeout": 300.0, "verify": False},
        )


# ── Palavras proibidas ─────────────────────────────────────────────────────────

PALAVRAS_PROIBIDAS = [
    "prefeitura", "municipio de", "município de", "camara municipal",
    "câmara municipal", "governo do estado", "governo do municipio",
    "governo federal", "estado de ", "estado do ", "fundacao municipal",
    "fundação municipal", "secretaria municipal", "secretaria de ",
    "tribunal de contas", "ministerio", "ministério", "camara dos deputados",
    "câmara dos deputados", "senado federal", "autarquia",
    "universidade federal", "universidade estadual",
]

# ── Diretórios ─────────────────────────────────────────────────────────────────

timestamp_for_files = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "..", "data", "synthetic_dataset_multi_company")
INPUT_FOLDER = os.environ.get("INPUT_FOLDER", _DATA_DIR)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_BASE_DIR, "..", "output", "methodology_II"),
)
OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, "extracao_{}_{}_filtrado.json".format(CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files))
GOLD_JSON_FILE = os.environ.get(
    "GOLD_JSON_FILE",
    os.path.join(_DATA_DIR, "gold_multi_company.json"),
)
REPORT_FILE = os.path.join(OUTPUT_DIR, "relatorio_{}_{}_filtrado.txt".format(CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files))
REPORT_FILE_JSON = os.path.join(OUTPUT_DIR, "relatorio_{}_{}_filtrado.json".format(CONFIG_PARAMS["MODELO"].replace(":", "_"), timestamp_for_files))
CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    os.environ.get("CHECKPOINT_NAME", "checkpoint.json"),
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Checkpoint carregado: {len(data.get('documentos', {}))} documento(s) já processado(s)")
        return data
    return None


def save_checkpoint(all_results):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    processed = len(all_results.get("documentos", {}))
    print(f"  [Checkpoint salvo: {processed} documento(s)]")


# ── Schemas Pydantic (API v2) ──────────────────────────────────────────────────

class Empresa(BaseModel):
    """Informações sobre uma empresa investigada. CNPJ é obrigatório."""
    nome: str = Field(description="Nome completo da empresa")
    cnpj: str = Field(description="CNPJ da empresa (obrigatório - não pode estar vazio)")

    @field_validator('cnpj')
    @classmethod
    def cnpj_nao_vazio(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("CNPJ não pode estar vazio")
        return v.strip()

    @field_validator('nome')
    @classmethod
    def nome_nao_vazio(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Nome não pode estar vazio")
        return v.strip()


class ListaEmpresas(BaseModel):
    """Lista de empresas encontradas no documento."""
    empresas: List[Empresa] = Field(description="Lista com todas as empresas identificadas que possuem CNPJ")


class Socio(BaseModel):
    """Informações sobre um sócio, proprietário ou administrador. Deve ter CPF OU RG."""
    nome: str = Field(description="Nome completo do sócio")
    cpf: str = Field(default="", description="CPF do sócio (se disponível)")
    rg: str = Field(default="", description="RG do sócio (se disponível)")

    @field_validator('nome')
    @classmethod
    def nome_nao_vazio(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Nome não pode estar vazio")
        return v.strip()

    @model_validator(mode='after')
    def verificar_documento(self) -> 'Socio':
        valores_invalidos = {"", "n/a", "não disponível", "não informado", "-", "na", "nd"}
        cpf_limpo = self.cpf.strip().lower() if self.cpf else ""
        rg_limpo = self.rg.strip().lower() if self.rg else ""
        cpf_valido = cpf_limpo and cpf_limpo not in valores_invalidos
        rg_valido = rg_limpo and rg_limpo not in valores_invalidos
        if not cpf_valido and not rg_valido:
            raise ValueError(f"Sócio {self.nome} deve ter pelo menos CPF ou RG válido")
        return self


class ListaSocios(BaseModel):
    """Lista de sócios de uma empresa específica. Sócios devem ter CPF OU RG."""
    socios: List[Socio] = Field(description="Lista com sócios que possuem CPF OU RG identificados")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _config_flag(key: str) -> bool:
    """Retorna True quando o parâmetro de CONFIG_PARAMS está definido como 'sim'."""
    return str(CONFIG_PARAMS.get(key, "não")).strip().lower() == "sim"


AUX_TERMS_EMPRESA = """Para auxiliar na identificação das empresas, procure pelos termos:
        - Pessoa jurídica de direito privado
        - Sociedade empresária limitada
        - Sociedade Limitada
        - LTDA
        - Microempresa
        - ME.
        - Empresa Individual de Responsabilidade Limitada
        - EIRELI
        - Sociedade Anônima
        - S.A.
        - Fundação
        - EPP"""

AUX_TERMS_SOCIOS_GENERIC = """Para auxiliar na identificação dos sócios desta empresa específica procure pelos termos:
        - Sócio-administrador da empresa
        - Sócia da empresa
        - Representada pelos sócios administradores
        - Representada pelo seu sócio gerente
        - Por seu Presidente
        - Seus respectivos sócios
        - De propriedade de
        - Representada por seu sócio-administrador
        - Representada por sua proprietária e administradora
        - Representado por seu proprietário e administrador
        - Representado pelos sócios
        - Representado por
        - Seu representante legal
        - Por seus sócios administradores
        - Administrador da sociedade"""

NEGATIVE_EMPRESA = """ATENÇÃO:
        - NÃO INVENTE informações
        - Só extraia empresas cujo nome ou CNPJ aparecem textualmente no contexto
        - NÃO inclua órgãos públicos (prefeituras, secretarias, autarquias, universidades públicas, etc.) como empresas
        - Se você não souber a resposta com base no contexto, não invente entidades"""

NEGATIVE_SOCIOS = """ATENÇÃO:
        - O sócio deve ter PELO MENOS CPF ou RG para ser incluído
        - NÃO INVENTE dados. Se o CPF ou RG não estiver no documento, deixe vazio
        - Se o sócio não tiver NENHUM documento (nem CPF nem RG), NÃO o inclua
        - Se você não souber a resposta com base no contexto, não invente entidades"""


def _aux_terms_socios(empresa_nome: str) -> str:
    """Termos auxiliares de sócios; enhancement adiciona frases com o nome da empresa."""
    if not _config_flag("AUX_TERMS"):
        return ""

    lines = [AUX_TERMS_SOCIOS_GENERIC]
    if _config_flag("PROMPT_ENHANCEMENT"):
        lines.append(f"""        - Sócio da {empresa_nome}
        - São proprietários da {empresa_nome}
        - Gerente da empresa {empresa_nome}
        - Diretor da empresa {empresa_nome}""")
    return "\n        \n".join(lines)


def build_question_empresa() -> str:
    parts = [
        """Você é um assistente encarregado de extrair informações de documentos investigativos.
        Identifique e extraia APENAS as empresas mencionadas no texto que POSSUEM CNPJ identificável, incluindo:
        - Nome completo das empresas
        - CNPJ das empresas (OBRIGATÓRIO)""",
    ]
    if _config_flag("AUX_TERMS"):
        parts.append(AUX_TERMS_EMPRESA)
    parts.append("E procure por CNPJs no formato: XX.XXX.XXX/XXXX-XX ou sequências de 14 números.")
    if _config_flag("NEGATIVE_PROMPT"):
        parts.append(NEGATIVE_EMPRESA)
    return "\n        \n".join(parts)


def build_question_socios(empresa_nome: str, empresa_referencia: str) -> str:
    parts = [
        f"""Você é um assistente encarregado de extrair informações de documentos investigativos.

        IMPORTANTE: Você deve extrair sócios que possuem PELO MENOS UM documento de identificação (CPF OU RG).
        Se um sócio tiver apenas CPF, inclua. Se tiver apenas RG, inclua. Se tiver ambos, melhor ainda.

        Identifique e extraia os sócios, proprietários e administradores da empresa "{empresa_referencia}" que atendam aos critérios:
        - Nome completo do sócio (OBRIGATÓRIO)
        - CPF do sócio (se disponível no documento)
        - RG do sócio (se disponível no documento)

        - Procure por CPFs no formato XXX.XXX.XXX-XX ou sequências de 11 números
        - Procure por RGs com números seguidos de órgão emissor (SSP, SSP/SC, etc.)""",
    ]
    aux = _aux_terms_socios(empresa_nome)
    if aux:
        parts.append(aux)
    if _config_flag("NEGATIVE_PROMPT"):
        parts.append(NEGATIVE_SOCIOS)
    parts.append(
        f'REGRA FINAL: Extraia APENAS os sócios da empresa "{empresa_referencia}" que possuem CPF OU RG identificados.'
    )
    return "\n        \n".join(parts)


def valid_company(nome_empresa):
    nome_lower = nome_empresa.lower()
    for palavra_proibida in PALAVRAS_PROIBIDAS:
        if palavra_proibida in nome_lower:
            return False
    return True


def create_retriever(vector_store, k):
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_rag_context(retriever, question):
    max_chars = CONFIG_PARAMS.get("MAX_CONTEXT_CHARS", 30000)
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    if len(context) > max_chars:
        print(f"  ⚠ Contexto RAG truncado: {len(context)} → {max_chars} caracteres")
        context = context[:max_chars]
    return context


# ── Extração ───────────────────────────────────────────────────────────────────

def extract_companies(retriever, llm):
    question_empresa = build_question_empresa()
    context = get_rag_context(retriever, question_empresa)

    prompt = f"""Com base no seguinte contexto extraído do documento, responda a pergunta.

Contexto: {context}

{question_empresa}"""

    empresas_filtradas = []
    empresas_rejeitadas = []

    try:
        result = invoke_structured_json(llm, prompt, ListaEmpresas)

        for empresa in result.empresas:
            if empresa.cnpj and empresa.cnpj.strip():
                if valid_company(empresa.nome):
                    empresas_filtradas.append({"nome": empresa.nome, "cnpj": empresa.cnpj})
                else:
                    empresas_rejeitadas.append({"nome": empresa.nome, "cnpj": empresa.cnpj, "motivo": "Entidade pública"})

        print(f"  ✓ Empresas privadas extraídas com CNPJ: {len(empresas_filtradas)}")

        if empresas_rejeitadas:
            print(f"  ⚠ Empresas rejeitadas (entidades públicas): {len(empresas_rejeitadas)}")
            for rej in empresas_rejeitadas:
                print(f"     - {rej['nome']} ({rej['motivo']})")

        return empresas_filtradas

    except Exception as e:
        print(f"  ⚠ Erro na extração estruturada: {e}")
        print(f"  ℹ Retornando lista de empresas privadas válidas: {len(empresas_filtradas)}")
        return empresas_filtradas


def extract_partners_for_company(retriever, llm, empresa_nome, empresa_cnpj=""):
    empresa_referencia = f"{empresa_nome} (CNPJ: {empresa_cnpj})" if empresa_cnpj else empresa_nome
    question_socios = build_question_socios(empresa_nome, empresa_referencia)
    context = get_rag_context(retriever, question_socios)

    intro = f"Com base no seguinte contexto extraído do documento, identifique os sócios da empresa {empresa_referencia} que possuem CPF OU RG."
    if _config_flag("NEGATIVE_PROMPT"):
        intro += "\n\nATENÇÃO: Inclua sócios que tenham PELO MENOS um dos documentos (CPF ou RG). Não é necessário ter ambos."

    prompt = f"""{intro}

Contexto: {context}

{question_socios}"""

    try:
        result = invoke_structured_json(llm, prompt, ListaSocios)

        socios = []
        socios_rejeitados = 0

        for socio in result.socios:
            cpf_valido = (socio.cpf and socio.cpf.strip() and
                         socio.cpf.strip().lower() not in ["n/a", "não disponível", "não informado", "-", ""])
            rg_valido = (socio.rg and socio.rg.strip() and
                        socio.rg.strip().lower() not in ["n/a", "não disponível", "não informado", "-", ""])

            if cpf_valido or rg_valido:
                socios.append({
                    "nome": socio.nome,
                    "cpf": socio.cpf if cpf_valido else "",
                    "rg": socio.rg if rg_valido else ""
                })
            else:
                socios_rejeitados += 1
                print(f"      ⚠ Sócio rejeitado (sem CPF nem RG): {socio.nome}")

        if socios_rejeitados > 0:
            print(f"      ℹ {socios_rejeitados} sócio(s) rejeitado(s) por falta de documentos")

        return socios

    except Exception as e:
        print(f"    ⚠ Erro na extração de sócios: {e}")
        return []


# ── Saída ──────────────────────────────────────────────────────────────────────

def save_final_json(all_results):
    final_output = {}

    for filename, doc_data in all_results["documentos"].items():
        empresas_list = []

        for empresa in doc_data["empresas"]:
            socios_formatados = []
            for socio in empresa["socios"]:
                if socio.get("cpf") or socio.get("rg"):
                    socios_formatados.append({
                        "nome": socio["nome"],
                        "cpf": socio.get("cpf", ""),
                        "rg": socio.get("rg", "")
                    })

            empresa_formatada = {
                "empresa": empresa["nome"],
                "cnpj": empresa["cnpj"],
                "socios": socios_formatados
            }
            empresas_list.append(empresa_formatada)

        final_output[filename] = empresas_list

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ JSON final salvo: {OUTPUT_JSON_FILE}")
    return OUTPUT_JSON_FILE


# ── Processamento ──────────────────────────────────────────────────────────────

def _truncate_text(text, max_chars):
    if max_chars and text and len(text) > max_chars:
        print(f"  ⚠ Texto truncado: {len(text)} → {max_chars} caracteres")
        return text[:max_chars]
    return text


def process_document(filename):
    file_path = os.path.join(INPUT_FOLDER, filename)

    print(f"\n=== Processando: {filename} ===")

    max_txt = CONFIG_PARAMS.get('MAX_TXT_CHARS', 50000)
    try:
        text = load_single_document(file_path, CONFIG_PARAMS['PAGES'], max_txt)
    except TypeError:
        text = load_single_document(file_path, CONFIG_PARAMS['PAGES'])
        text = _truncate_text(text, max_txt)
    if not text:
        print(f"Erro: Não foi possível carregar {filename}")
        return None

    chunks = split_text(
        text,
        CONFIG_PARAMS['CHUNK_SIZE'],
        CONFIG_PARAMS['CHUNK_OVERLAP'],
        CONFIG_PARAMS['SEPARATORS']
    )

    # Embeddings usam o servidor Ollama (endpoint /api/embeddings funciona)
    vector_store = create_embeddings(chunks, CONFIG_PARAMS['EMBEDDING_MODEL'], base_url=OLLAMA_HOST)
    retriever = create_retriever(vector_store, CONFIG_PARAMS['K'])

    # LLM de chat usa OpenRouter (endpoint de chat do Ollama retorna 404)
    llm = create_llm()

    print("--- Identificando empresas PRIVADAS (com CNPJ) ---")
    empresas = extract_companies(retriever, llm)

    if not empresas:
        print("  ⚠ Nenhuma empresa privada com CNPJ foi encontrada neste documento")
        delete_vector_store(vector_store)
        return {
            "arquivo": filename,
            "timestamp_processamento": datetime.datetime.now().isoformat(),
            "total_empresas": 0,
            "total_socios": 0,
            "empresas": [],
            "configuracao": CONFIG_PARAMS.copy(),
            "filtros_aplicados": "Empresas privadas com CNPJ | Sócios com CPF OU RG",
            "palavras_filtradas": PALAVRAS_PROIBIDAS
        }

    print(f"Empresas privadas encontradas (com CNPJ): {len(empresas)}")
    for i, empresa in enumerate(empresas, 1):
        print(f"  {i}. {empresa['nome']} - CNPJ: {empresa['cnpj']}")

    print("--- Buscando sócios (COM CPF OU RG) ---")

    for empresa in empresas:
        print(f"  Processando sócios de: {empresa['nome']}")
        socios = extract_partners_for_company(retriever, llm, empresa['nome'], empresa['cnpj'])
        empresa['socios'] = socios

        if socios:
            print(f"    ✓ Sócios encontrados COM documentos: {len(socios)}")
            for socio in socios:
                docs = []
                if socio.get('cpf'):
                    docs.append(f"CPF: {socio['cpf']}")
                if socio.get('rg'):
                    docs.append(f"RG: {socio['rg']}")
                print(f"      - {socio['nome']} | {' | '.join(docs)}")
        else:
            print(f"    ⚠ Nenhum sócio com CPF ou RG foi encontrado para esta empresa")

    delete_vector_store(vector_store)

    document_result = {
        "arquivo": filename,
        "timestamp_processamento": datetime.datetime.now().isoformat(),
        "total_empresas": len(empresas),
        "total_socios_com_documento": sum(len(empresa['socios']) for empresa in empresas),
        "empresas": empresas,
        "configuracao": CONFIG_PARAMS.copy(),
        "filtros_aplicados": "Empresas privadas com CNPJ | Sócios com CPF OU RG",
        "palavras_filtradas": PALAVRAS_PROIBIDAS
    }

    print(f"✓ Finalizado: {filename} - {document_result['total_empresas']} empresas privadas (com CNPJ), {document_result['total_socios_com_documento']} sócios (com CPF ou RG)")

    return document_result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("="*60)
    print("EXTRATOR DE EMPRESAS E SÓCIOS - V10")
    print(f"PROVIDER: {PROVIDER.upper()}")
    print("FILTROS APLICADOS:")
    print("  - Empresas: Apenas PRIVADAS com CNPJ")
    print("  - Sócios: Devem ter CPF OU RG")
    print("MODIFICADORES DE PROMPT:")
    print(f"  - Termos auxiliares (AUX_TERMS): {CONFIG_PARAMS['AUX_TERMS']}")
    print(f"  - Prompt enhancement (PROMPT_ENHANCEMENT): {CONFIG_PARAMS['PROMPT_ENHANCEMENT']}")
    print(f"  - Prompt negativo (NEGATIVE_PROMPT): {CONFIG_PARAMS['NEGATIVE_PROMPT']}")
    print("="*60)
    print(f"Configuração: {CONFIG_PARAMS}")
    print(f"Diretório de entrada: {INPUT_FOLDER}")
    print("\n⚠ IMPORTANTE:")
    print("  - Apenas empresas PRIVADAS com CNPJ serão extraídas")
    print("  - Entidades públicas serão FILTRADAS (palavras proibidas):")
    for palavra in PALAVRAS_PROIBIDAS:
        print(f"    • {palavra}")
    print("  - Sócios devem ter pelo menos CPF ou RG")

    checkpoint = load_checkpoint()

    if checkpoint and checkpoint.get("documentos"):
        all_results = checkpoint
        all_results["processamento"]["retomado_em"] = datetime.datetime.now().isoformat()
        already_done = set(all_results["documentos"].keys())
        print(f"\n>>> RETOMANDO de checkpoint: {len(already_done)} documento(s) já processado(s)")
        for fn in sorted(already_done):
            print(f"    - {fn} (já processado)")
    else:
        all_results = {
            "processamento": {
                "inicio": datetime.datetime.now().isoformat(),
                "configuracao": CONFIG_PARAMS,
                "ollama_host": OLLAMA_HOST,
                "provider": PROVIDER,
                "filtros": {
                    "empresas": "Apenas privadas com CNPJ (excluindo entidades públicas)",
                    "socios": "Deve ter CPF ou RG (pelo menos um)",
                    "palavras_filtradas": PALAVRAS_PROIBIDAS
                }
            },
            "documentos": {},
            "estatisticas_filtragem": {
                "total_empresas_rejeitadas": 0,
                "empresas_rejeitadas_detalhes": []
            }
        }
        already_done = set()

    processed_count = len(already_done)
    all_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                        if f.endswith(".pdf") or f.endswith(".txt")])
    remaining = [f for f in all_files if f not in already_done]
    print(f"\nArquivos restantes: {len(remaining)} de {len(all_files)}")

    for i, filename in enumerate(remaining):
        try:
            document_result = process_document(filename)
            if document_result:
                all_results["documentos"][filename] = document_result
                processed_count += 1
                save_checkpoint(all_results)
            if i < len(remaining) - 1:
                print("  Aguardando 10s antes do próximo documento...")
                time.sleep(10)
        except Exception as e:
            print(f"\n  ERRO ao processar {filename}: {e}")
            print(f"  Salvando checkpoint e aguardando 60s antes de continuar...")
            save_checkpoint(all_results)
            time.sleep(60)
            continue

    elapsed = time.time() - start_time
    all_results["processamento"]["fim"] = datetime.datetime.now().isoformat()
    all_results["processamento"]["tempo_execucao_segundos"] = elapsed
    all_results["processamento"]["documentos_processados"] = processed_count

    total_empresas = sum(doc["total_empresas"] for doc in all_results["documentos"].values())
    total_socios = sum(doc.get("total_socios_com_documento", 0) for doc in all_results["documentos"].values())

    all_results["estatisticas"] = {
        "total_documentos": processed_count,
        "total_empresas_privadas_com_cnpj": total_empresas,
        "total_socios_com_cpf_ou_rg": total_socios
    }

    print("\n" + "="*60)
    print("RESUMO FINAL")
    print("="*60)
    print(f"Documentos processados: {processed_count}")
    print(f"Total de empresas PRIVADAS extraídas (com CNPJ): {total_empresas}")
    print(f"Total de sócios extraídos (com CPF ou RG): {total_socios}")
    print(f"Tempo de execução: {elapsed:.2f}s ({elapsed/60:.2f}min)")

    print("\nDetalhamento por documento:")
    for filename, doc_data in all_results["documentos"].items():
        total_socios_doc = doc_data.get("total_socios_com_documento", 0)
        print(f"  📄 {filename}:")
        print(f"     - Empresas privadas (com CNPJ): {doc_data['total_empresas']}")
        print(f"     - Sócios (com CPF ou RG): {total_socios_doc}")

    results_json = save_final_json(all_results)

    print("\n" + "="*60)
    print("VALIDAÇÃO DOS RESULTADOS GERADOS".center(60))
    print("="*60)

    validador = ValidadorDocumentos(
        arquivo_gabarito=GOLD_JSON_FILE,
        arquivo_resposta=results_json,
        config_params=CONFIG_PARAMS
    )
    executou = validador.executar(REPORT_FILE, REPORT_FILE_JSON)

    if executou:
        print("\n✓ Validação concluída com sucesso!")
    else:
        print("\n✗ Erro durante a validação!")
        return 1

    return all_results


if __name__ == "__main__":
    results = main()

    print("\n" + "="*60)
    print("ARQUIVO JSON GERADO:")
    print(f"  {OUTPUT_JSON_FILE}")
    print("\n⚠ FILTROS APLICADOS:")
    print("  - Apenas empresas PRIVADAS com CNPJ foram extraídas")
    print("  - Entidades públicas foram FILTRADAS automaticamente")
    print("  - Apenas sócios com CPF OU RG identificados foram extraídos")
    print("\nResultados também disponíveis no dicionário 'results'")
    print("Execução finalizada!")