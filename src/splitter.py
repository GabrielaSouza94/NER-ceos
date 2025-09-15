from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text,size, overlap, separator):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=separator,
    )
    chunks = splitter.split_text(text)
    print(f" Total de chunks: {len(chunks)}")
    return chunks

