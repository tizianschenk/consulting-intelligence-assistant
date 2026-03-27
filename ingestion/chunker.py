"""
ingestion/chunker.py

Splits LangChain Document objects into smaller chunks that are small enough
to embed and retrieve precisely, but large enough to preserve meaning.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""RecursiveCharacterTextSplitter is LangChain's smartest built-in text splitter. 
The word "recursive" is the key — it doesn't just cut at a fixed character count. 
Instead it tries to split at the most natural break point it can find, in this priority order:

First it tries to split on paragraph breaks (\n\n)
If the chunk is still too big, it splits on line breaks (\n)
If still too big, it splits on spaces (between words)
Last resort: it cuts at the exact character limit

This means it always tries to keep paragraphs together, then sentences, then words, 
only cutting mid-word if absolutely necessary. 
That's the "recursive" part — it keeps trying smaller separators until the chunk fits."""


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks suitable for embedding.

    Args:
        documents:    list of Documents from the PDF loader
        chunk_size:   maximum characters per chunk (default 800)
        chunk_overlap: characters shared between adjacent chunks (default 150)

    Returns:
        a new, longer list of smaller Document objects
    """

    # Create the splitter with our chosen parameters.
    # separators defines the priority order for where to cut —
    # try paragraph breaks first, then line breaks, then spaces, then any character.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,  # use character count (not token count) as the measure
    )

    # split_documents() takes our list of Documents and returns a new list
    # of smaller Documents. Crucially, it PRESERVES the metadata from the
    # original Document — so each chunk still knows which file and page it
    # came from. That's what keeps our citations working after chunking.
    chunks = splitter.split_documents(documents)

    # Add a chunk index to each chunk's metadata so we know its position
    # within the original document. This helps with debugging and citations.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"  ✅  Chunked {len(documents)} pages → {len(chunks)} chunks")
    print(f"       chunk_size={chunk_size}, overlap={chunk_overlap}")

    return chunks


def inspect_chunks(chunks: list[Document], n: int = 3) -> None:
    """
    Print the first n chunks so you can visually verify the chunking quality.
    Use this during development to check that chunks make sense.

    Args:
        chunks: list of chunk Documents
        n:      how many chunks to display (default 3)
    """
    print(f"\n{'=' * 60}")
    print(f"CHUNK INSPECTION — showing {min(n, len(chunks))} of {len(chunks)} chunks")
    print(f"{'=' * 60}")

    for i, chunk in enumerate(chunks[:n]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Characters : {len(chunk.page_content)}")
        print(f"Source     : {chunk.metadata.get('file_name', 'unknown')}")
        print(f"Page       : {chunk.metadata.get('page', '?')}")
        print("Text preview:")
        # Show the first 200 chars with a visual cutoff marker
        print(f"  {chunk.page_content[:200]}...")
        print()
