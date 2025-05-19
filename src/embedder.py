from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_embeddings(chunks):
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        chunks,
        embedding_model,
        collection_name="temp_collection"
        # sem persist_directory = uso em memória
    )
    print("Embeddings criados em memória com sucesso.")
    return vector_store