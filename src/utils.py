import shutil
import os

def delete_vector_store(persist_directory):
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree (persist_directory)
            print (f" Vector store apagado: {persist_directory}")
        except PermissionError:
            print (f" Não foi possível apagar. Arquivos ainda estão em uso: {persist_directory}")
        else:
            print (f" Nenhum vector store encontrado em: {persist_directory}")
