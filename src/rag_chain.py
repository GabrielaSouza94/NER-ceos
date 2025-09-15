from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
def run_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k":20})
    llm = ChatOpenAI(temperature=0.1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
