import shutil
import os
import gc

def delete_vector_store(vector_store=None, persist_path=None):
    try:
        if vector_store is not None:
            vector_store.delete_collection ()
            del vector_store

        gc.collect ()

        if persist_path and os.path.exists (persist_path):
            shutil.rmtree (persist_path)
            print (f"[✔] Vector store em '{persist_path}' apagado com sucesso.")
        else:
            print (f"[✔] Vector store removido da memória. Nada a limpar em disco.")
        return True
    except Exception as e:
        print (f"[✖] Erro ao limpar vector store: {e}")
        return False