from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def run_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    llm = ChatOpenAI(temperature=0.4)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
