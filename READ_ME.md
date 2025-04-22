# RAG Pipeline

Pipeline simples de RAG (Retrieval-Augmented Generation) usando LangChain, OpenAI e ChromaDB.
##Estrutura

- `src/`: Códigos fonte 
- `input/`: PDFs usados para o RAG
- `embeddings/`: Armazenamento local dos vetores
- `.env/`: SET aqui a chave da open ai

##Como rodar

1. Instale as dependências:
```bash
pip install -r requirements.txt

2. Execute : 
```bash
python src/main.py