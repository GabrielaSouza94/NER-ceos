from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f" Total de chunks: {len(chunks)}")
    return chunks
