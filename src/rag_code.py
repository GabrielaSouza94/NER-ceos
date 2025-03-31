import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from src.utils import read_pdf

# Carregar variáveis de ambiente
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("A chave da OpenAI não foi encontrada. Verifique o arquivo .env")

# 1. Ler PDF
document_path = "input_files/termo_prestacao_servicos_exemplo.pdf"
text = read_pdf(document_path)

# 2. Dividir texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_text(text)
print(f"Total de chunks gerados: {len(chunks)}")

# 3. Gerar embeddings e persistir com Chroma
embedding_model = OpenAIEmbeddings()
persist_directory = 'embeddings/chroma-openai/'

vector_store = Chroma.from_texts(
    chunks,
    embedding_model,
    persist_directory=persist_directory
)
vector_store.persist()
print("\n✅ Embeddings salvos com sucesso.")