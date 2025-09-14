from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        add_start_index=True,
        separators=[";"],
    )
    chunks = splitter.split_text(text)
    print(f" Total de chunks: {len(chunks)}")
    return chunks
