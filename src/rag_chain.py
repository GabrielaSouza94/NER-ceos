from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
def run_qa_chain(vector_store,k,temperature):
    retriever = vector_store.as_retriever(search_kwargs={"k":k})
    llm = ChatOpenAI(temperature=temperature)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
