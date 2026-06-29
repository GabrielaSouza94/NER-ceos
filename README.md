# NER-ceos

Code and data for the paper **"Micro-Prompt Chaining for Relational Extraction:
An Inference Orchestration Architecture for Complex Documents"**, developed
within the CEOS project (master's research in data).

The work addresses **relational entity extraction** (companies and their
shareholders) from long, unstructured legal-administrative documents. We propose
an inference-orchestration architecture based on **Micro-Prompt Chaining**, which
decomposes extraction into dependent, schema-constrained inferences with
task-scoped retrieval and Pydantic validation, and compare it against a baseline
RAG pipeline.

## Repository structure

```
NER-ceos/
├── README.md
├── requirements.txt
└── reviewer_package/        # Self-contained, publicly reproducible package
    ├── README.md            #   → detailed setup and run instructions
    ├── requirements.txt
    ├── code/                #   pipelines + validators + helpers
    └── data/                #   synthetic single-company & multi-company corpora
```

## Reviewer package

The [`reviewer_package/`](reviewer_package/) directory contains everything needed
to reproduce the **synthetic-corpus** experiments without any confidential data:

- **Methodology I** — baseline linear RAG pipeline.
- **Methodology II** — the proposed Micro-Prompt Chaining architecture.
- **Synthetic datasets** — `single company` and `multi company` corpora (200
  procedurally generated documents in total, with gold answers). All identifiers
  are fictitious.

See [`reviewer_package/README.md`](reviewer_package/README.md) for installation,
environment variables, and how to run each methodology.

> The confidential corpus provided by the Brazilian Public Ministry is **not**
> part of this repository.

## Requirements

```bash
pip install -r reviewer_package/requirements.txt
```

No credentials are hard-coded. When using a cloud LLM provider, supply the API
key via the `OPENROUTER_API_KEY` environment variable.
