from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def run_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k":20})
    llm = ChatOpenAI(temperature=0.1)

    try:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    except Exception:
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question. If you don't know, say you don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Accept string input directly and return a legacy-compatible output shape
    # {'query': str, 'result': str, 'source_documents': list[Document]}
    rag_chain_legacy = (
        {"input": RunnablePassthrough()}
        | rag_chain
        | RunnableLambda(lambda x: {
            "query": x.get("input"),
            "result": x.get("answer"),
            "source_documents": x.get("context"),
        })
    )
    return rag_chain_legacy
