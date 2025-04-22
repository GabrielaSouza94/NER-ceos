from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_embeddings(chunks, persist_directory='embeddings/chroma-openai/'):
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        chunks,
        embedding_model,
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(" Embeddings salvos com sucesso.")
    return vector_store
