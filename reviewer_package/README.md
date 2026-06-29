# Reviewer Package — Micro-Prompt Chaining for Relational Extraction

This package contains the code and the **publicly reproducible synthetic datasets**
needed to reproduce the synthetic-corpus experiments reported in the paper
*"Micro-Prompt Chaining for Relational Extraction: An Inference Orchestration
Architecture for Complex Documents"*.

> The confidential corpus provided by the Brazilian Public Ministry is **not**
> included here. Only the procedurally generated synthetic corpus, which can be
> shared openly, is distributed in this package.

## Contents

```
reviewer_package/
├── README.md
├── requirements.txt
├── code/
│   ├── pipeline_methodology_I.py    # Methodology I  — baseline linear RAG (text/CSV, no schema)
│   ├── pipeline_methodology_II.py   # Methodology II — Micro-Prompt Chaining (schema-constrained, Pydantic)
│   ├── parse_resposta_v1.py         # response parser used by Methodology I
│   ├── val3.py                      # validator used by Methodology I  (entity/ID matching)
│   ├── validador_json3.py           # validator used by Methodology II (JSON document matching)
│   ├── loader.py                    # document loader (PDF/TXT)
│   ├── splitter.py                  # recursive text splitter (chunking)
│   ├── embedder.py                  # embeddings (local SentenceTransformers or Ollama)
│   └── utils_ollama.py              # helpers (vector-store cleanup, text sanitization)
└── data/
    ├── synthetic_dataset_single_company/   # 100 single-company synthetic documents + gold answers
    │   ├── doc_grupo1_001.txt ... doc_grupo1_100.txt
    │   ├── gold_single_company.csv
    │   └── gold_single_company.json
    └── synthetic_dataset_multi_company/    # 100 multi-company synthetic documents + gold answers
        ├── doc_grupo2_101.txt ... doc_grupo2_200.txt
        ├── gold_multi_company.csv
        └── gold_multi_company.json
```

### Datasets

Both datasets are **fully synthetic** (procedurally generated). All names,
company names, CNPJ, CPF and RG identifiers are fictitious and do not refer to
any real person or company.

- **synthetic dataset — single company**: documents describing a single company
  and its shareholders.
- **synthetic dataset — multi company**: documents describing multiple companies
  (and their respective shareholders) per document.

### Methodologies

- **Methodology I** (`pipeline_methodology_I.py`): the baseline pipeline, which
  performs document-level retrieval and free-text extraction (no schema
  constraint), prone to *contextual fragmentation*.
- **Methodology II** (`pipeline_methodology_II.py`): the proposed
  **Micro-Prompt Chaining** architecture, which decomposes relational extraction
  into dependent, schema-constrained inferences (Pydantic) with task-scoped
  retrieval.

## Setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration (environment variables)

No credentials are hard-coded. The pipelines are configured through environment
variables (sensible defaults are provided for everything except the API key).

| Variable             | Description                                                        | Default                                   |
| -------------------- | ------------------------------------------------------------------ | ----------------------------------------- |
| `PROVIDER`           | `openrouter` (cloud LLM) or `ollama` (local/self-hosted LLM)       | `ollama` (Method. I) / `openrouter` (II)  |
| `OPENROUTER_API_KEY` | API key, **required** when `PROVIDER=openrouter`                   | *(empty — must be supplied)*              |
| `OLLAMA_HOST`        | Base URL of the Ollama server (used for `ollama` and embeddings)   | `http://localhost:11434`                  |
| `INPUT_FOLDER`       | Folder with the input documents                                    | bundled synthetic dataset                 |
| `OUTPUT_DIR`         | Folder where results/reports are written                           | `../output/methodology_{I,II}`            |
| `GOLD_CSV_FILE`      | Gold answers (CSV) for Methodology I                               | bundled `gold_single_company.csv`         |
| `GOLD_JSON_FILE`     | Gold answers (JSON) for Methodology II                             | bundled `gold_multi_company.json`         |
| `CONFIG_JSON`        | JSON string overriding `CONFIG_PARAMS` (model, chunking, K, etc.)  | *(unset)*                                 |

## Running

From the `reviewer_package/code/` directory:

```bash
# Methodology II (Micro-Prompt Chaining) on the multi-company dataset via OpenRouter
export PROVIDER=openrouter
export OPENROUTER_API_KEY=<your_key>
python pipeline_methodology_II.py

# Methodology I (baseline) on the single-company dataset
python pipeline_methodology_I.py
```

To point a methodology at the other dataset, override `INPUT_FOLDER` and the
corresponding `GOLD_*` variable, e.g.:

```bash
export INPUT_FOLDER=../data/synthetic_dataset_single_company
export GOLD_JSON_FILE=../data/synthetic_dataset_single_company/gold_single_company.json
python pipeline_methodology_II.py
```

The evaluated open-weight models in the paper are Llama 3.3, DeepSeek-R1, Qwen3
and Gemma 3; the model is selected via `CONFIG_PARAMS["MODELO"]` (overridable
through `CONFIG_JSON`).
