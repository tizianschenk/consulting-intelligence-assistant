"""
rag/retriever.py

The bridge between the vector store and the chain

Retrieves the most relevant document chunks from ChromaDB for a given query.
This sits between the user's question and the LLM — it fetches the evidence
the LLM will use to construct a grounded, cited answer.

This file's job is simple but important: take a user's question, go find the most relevant chunks from ChromaDB,
and return them in a clean format ready to be injected into a prompt.
It's a thin layer on top of the VectorStore class we already built — but separating it out keeps our code organised.
The vector store handles how to search; the retriever handles when and what to search for.
"""

from langchain_core.documents import Document

from ingestion.embedder import VectorStore

"""Just two imports. 
We bring in Document (our standard container) and VectorStore (our ChromaDB wrapper from Step 2). 
The retriever doesn't need to know anything about embeddings or ChromaDB directly — 
it just asks the VectorStore to do the searching.
This is a design principle called separation of concerns — 
each file has one job and doesn't reach into another file's territory."""


class Retriever:
    """
    Wraps the VectorStore to provide document retrieval for the RAG chain.

    Responsible for:
    - Accepting a natural language query
    - Fetching the top-k most relevant chunks
    - Optionally filtering results below a similarity threshold
    - Formatting chunks for injection into the LLM prompt
    """

    def __init__(
        self, vector_store: VectorStore, k: int = 4, min_similarity: float = 0.0
    ):
        """
        Args:
            vector_store:   an initialised VectorStore instance
            k:              how many chunks to retrieve per query (default 4)
            min_similarity: minimum similarity score to include a chunk (default 0.0)
                            set higher (e.g. 0.3) to filter out weak matches
        """

        # Store the vector store and settings as instance variables
        # so every method in this class can access them via self.
        self.vector_store = vector_store
        self.k = k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> list[Document]:
        """
        Find the most relevant chunks for a given query.

        Args:
            query: the user's natural language question

        Returns:
            list of Document objects, most relevant first,
            filtered by minimum similarity threshold
        """

        # Ask the vector store to do the actual similarity search.
        # It converts the query to a vector and finds the k closest chunks.
        results = self.vector_store.search(query, n_results=self.k)

        # Filter out chunks below our minimum similarity threshold.
        # This prevents the LLM from receiving totally irrelevant context
        # just because it was the "least bad" match in an empty or unrelated store.
        # For example, if someone asks about "climate change" but we only have
        # a CV in our store, we don't want to return random CV chunks as "context".
        filtered = [
            doc
            for doc in results
            if doc.metadata.get("similarity_score", 0) >= self.min_similarity
        ]

        if not filtered:
            print(
                f"  ⚠️  No chunks met the similarity threshold ({self.min_similarity})"
            )
        else:
            print(
                f"  ✅  Retrieved {len(filtered)} chunks (top score: "
                f"{filtered[0].metadata.get('similarity_score', '?')})"
            )

        return filtered

    def format_context(self, docs: list[Document]) -> str:
        """
        Format a list of retrieved Documents into a single context string
        that gets injected into the LLM prompt.

        Each chunk is formatted with its text and a source citation tag.
        The LLM is instructed to use these tags when citing sources.

        Example output:
            [Source: report.pdf | Page 3]
            "Digital transformation increases revenue by 20%..."

            [Source: report.pdf | Page 7]
            "Implementation requires significant change management..."

        Args:
            docs: list of retrieved Document objects

        Returns:
            a single formatted string ready for prompt injection
        """

        if not docs:
            return "No relevant context found in the uploaded documents."

        formatted_chunks = []
        for doc in docs:
            # Pull source metadata — use .get() with fallbacks in case
            # metadata keys are missing (defensive programming)
            file_name = doc.metadata.get("file_name", "Unknown source")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("similarity_score", "?")

            # Format each chunk as a labelled block.
            # The [Source: ...] tag is what the LLM will reference in its answer.
            chunk_text = (
                f"[Source: {file_name} | Page {page} | Relevance: {score}]\n"
                f"{doc.page_content}"
            )
            formatted_chunks.append(chunk_text)

        # Join all chunks with a clear separator so the LLM can distinguish
        # where one chunk ends and the next begins.
        return "\n\n---\n\n".join(formatted_chunks)
