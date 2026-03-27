"""
ingestion/embedder.py

Converts text chunks into vector embeddings using sentence-transformers,
then stores them in a persistent ChromaDB vector database.

This is the core of the RAG system's 'memory' — once documents are embedded
and stored here, the system can find relevant passages for any question.
"""

from pathlib import Path

# the vector database client. We use it to create, connect to, and query our local database
import chromadb

# a configuration object that controls ChromaDB's behaviour. We'll use it to tell ChromaDB where to save its files on disk.
from chromadb.config import Settings
from langchain_core.documents import Document

# the class that loads our embedding model (all-MiniLM-L6-v2) and converts text into vectors. This runs entirely locally — no internet call.
from sentence_transformers import SentenceTransformer

# Where ChromaDB will save its data on disk.
# Using Path ensures this works on Mac, Windows, and Linux.
# The folder 'data/chroma_db' will be created automatically if it doesn't exist.
CHROMA_PATH = Path("data/chroma_db")

# The name of our collection inside ChromaDB.
# A 'collection' is like a table in a regular database —
# a named group of stored vectors. We use one collection for all documents.
COLLECTION_NAME = "consulting_docs"

# The embedding model we're using.
# all-MiniLM-L6-v2 produces 384-dimensional vectors.
# It's small, fast, and runs locally — no API cost ever.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

"""A class in Python is a blueprint for an object that bundles together related data and functions. 
Our VectorStore class bundles together the embedding model, the ChromaDB client, and all the methods 
for adding and searching documents."""


class VectorStore:
    """
    Manages the embedding and storage of document chunks in ChromaDB.

    This class handles:
    - Loading the embedding model (once, on startup)
    - Connecting to the persistent ChromaDB database
    - Adding new document chunks
    - Searching for relevant chunks by similarity
    """

    def __init__(self):
        """
        __init__ is the constructor — it runs automatically when you create
        a VectorStore object with: store = VectorStore()

        Here we load the embedding model and connect to ChromaDB.
        Both are expensive operations (slow to start), so we do them
        once here rather than every time we embed or search.
        """

        print("  🔄  Loading embedding model...")

        # Load the sentence-transformer model into memory.
        # On first run this downloads ~90MB from HuggingFace (already cached
        # from our setup_check.py run, so it loads instantly now).
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"  ✅  Embedding model loaded: {EMBEDDING_MODEL}")
        print("  🔄  Connecting to ChromaDB...")

        # Create a persistent ChromaDB client.
        # 'persistent' means it saves data to disk at CHROMA_PATH —
        # your vectors survive when you close the app and reopen it.
        # Without persistence, everything would be lost on exit.
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
            # anonymized_telemetry=False: ChromaDB by default sends anonymous
            # usage stats to its developers. We turn this off for privacy.
        )

        # Get or create our collection.
        # get_or_create_collection: if the collection already exists (e.g. we've
        # run the app before), it opens it. If not, it creates a fresh one.
        # This means we can safely call this every time without wiping our data.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            # hnsw:space="cosine": this tells ChromaDB which mathematical formula
            # to use when measuring distance between vectors.
            # 'cosine' measures the ANGLE between two vectors rather than the
            # straight-line distance. For text embeddings, angle-based similarity
            # works better than raw distance because it ignores vector magnitude
            # and focuses purely on direction — which is what captures meaning.
        )

        print(f"  ✅  ChromaDB connected: collection '{COLLECTION_NAME}'")
        print(f"       Documents currently stored: {self.collection.count()}")

    def add_documents(self, chunks: list[Document]) -> None:
        """
        Embed a list of Document chunks and store them in ChromaDB.

        For each chunk we store:
        - the text (so we can retrieve and display it)
        - the embedding vector (so we can search by similarity)
        - the metadata (source, page — for citations)

        Args:
            chunks: list of Document objects from chunker.py
        """

        if not chunks:
            print("  ⚠️  No chunks provided — nothing to embed.")
            return

        print(f"  🔄  Embedding {len(chunks)} chunks...")

        # Extract just the text strings from our Document objects.
        # The embedding model works with plain strings, not Document objects.
        texts = [chunk.page_content for chunk in chunks]

        # Convert all texts to vectors in one batch call.
        # Batch processing is much faster than embedding one at a time —
        # the model processes all texts in parallel on your CPU (or GPU if available).
        # Each text becomes a list of 384 floats.
        # embeddings is now a 2D array: shape (num_chunks, 384)
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
        )
        # .tolist() converts the numpy array that encode() returns into a plain
        # Python list — which is what ChromaDB expects when storing embeddings.
        embeddings = embeddings.tolist()

        # Build the metadata list — one dict per chunk.
        # ChromaDB stores metadata alongside each vector so we can retrieve
        # the source and page number when we return results to the user.
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate a unique ID for each chunk.
        # ChromaDB requires every stored item to have a unique string ID.
        # We combine the filename and chunk index to make it unique and readable.
        ids = [
            f"{chunk.metadata.get('file_name', 'doc')}_chunk_{i}"
            for i, chunk in enumerate(chunks)
        ]

        # Store everything in ChromaDB.
        # upsert = "update if exists, insert if not"
        # This means re-running the ingestion with the same PDF won't create
        # duplicate entries — it'll just overwrite the existing ones.
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"  ✅  Stored {len(chunks)} chunks in ChromaDB")
        print(f"       Total documents in store: {self.collection.count()}")

    def search(self, query: str, n_results: int = 4) -> list[Document]:
        """
        Find the most relevant chunks for a given query.

        This is the retrieval half of RAG. We convert the query to a vector,
        then ask ChromaDB for the n_results closest stored vectors.

        Args:
            query:     the user's natural language question
            n_results: how many chunks to retrieve (default 4)

        Returns:
            list of Document objects, ordered by relevance (most relevant first)
        """

        # Convert the query string into a vector using the same model
        # we used to embed the documents. This is critical — you must use
        # the SAME model for both embedding and querying, or the vector spaces
        # won't match and similarity search will return nonsense.
        query_vector = self.embedding_model.encode(query).tolist()

        # Ask ChromaDB for the n most similar chunks.
        # It compares the query_vector against every stored vector using
        # cosine similarity, then returns the closest matches.
        results = self.collection.query(
            query_embeddings=[
                query_vector
            ],  # wrapped in a list (ChromaDB expects batch)
            n_results=min(n_results, self.collection.count()),
            # min() prevents errors when the collection has fewer items than n_results
            include=["documents", "metadatas", "distances"],
            # distances: how far each result is from the query (lower = more similar)
        )

        # Repackage ChromaDB's results back into LangChain Document objects.
        # ChromaDB returns parallel lists: results["documents"][0] is a list of texts,
        # results["metadatas"][0] is the corresponding list of metadata dicts.
        # We zip them together to pair each text with its metadata.
        retrieved_docs = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(
                page_content=text,
                metadata={
                    **metadata,  # unpack all existing metadata (source, page, etc.)
                    "similarity_score": round(1 - distance, 4),
                    # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite).
                    # We convert to similarity SCORE (1=identical, 0=opposite)
                    # by doing 1 - distance. More intuitive: higher = better match.
                },
            )
            retrieved_docs.append(doc)

        return retrieved_docs

    def get_document_count(self) -> int:
        """Returns the total number of chunks stored in ChromaDB."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Delete all stored vectors and start fresh.
        Useful during development when you want to re-index everything.
        """
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("  ✅  Vector store cleared.")
