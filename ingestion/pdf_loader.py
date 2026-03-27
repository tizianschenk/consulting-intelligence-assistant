"""
ingestion/pdf_loader.py

Responsible for reading PDF files and converting them into LangChain
Document objects that the rest of the pipeline can work with.
"""

from pathlib import Path  # modern Python way to work with file paths

import fitz  # this is PyMuPDF — "fitz" is its import name

# We'll try fitz first, and only switch to pdfplumber if fitz extracts very little text
# (which happens with some scanned or complex PDFs)
import pdfplumber  # our fallback for complex PDFs
from langchain_core.documents import Document  # the standard LangChain container


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a single PDF file and return a list of LangChain Documents.
    Each page becomes one Document object.

    Args:
        file_path: path to the PDF file (string or Path object)

    Returns:
        list of Document objects, one per page
    """

    # Convert to a Path object if a plain string was passed in.
    # This lets callers pass either "report.pdf" or Path("report.pdf") — both work.
    path = Path(file_path)

    # Safety check: does this file actually exist?
    # Better to catch this here with a clear message than get a cryptic error later.
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Safety check: is this actually a PDF?
    # .suffix gives us the file extension — e.g. ".pdf", ".docx", ".txt"
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    # Try PyMuPDF first (faster, handles most PDFs well)
    documents = _load_with_fitz(path)

    # If PyMuPDF extracted very little text, the PDF might be complex or
    # have unusual formatting. Fall back to pdfplumber which is slower
    # but better at extracting text from tricky layouts.
    total_text = sum(len(doc.page_content) for doc in documents)
    if total_text < 100:  # less than 100 characters total is a red flag
        print(
            f"  ⚠️  PyMuPDF got little text ({total_text} chars). Trying pdfplumber..."
        )
        documents = _load_with_pdfplumber(path)
        # The < 100 threshold is a heuristic — an educated guess based on experience. 100 characters is about one sentence.
        # If an entire PDF produced less than one sentence of text, something went wrong with extraction.

    print(f"  ✅  Loaded '{path.name}': {len(documents)} pages extracted")
    return documents


def _load_with_fitz(path: Path) -> list[Document]:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    This is our primary extraction method — fast and accurate for most PDFs.

    The leading underscore in _load_with_fitz signals that this is a
    'private' helper function — it's meant to be used only inside this file,
    not called directly from outside.
    """
    documents = []

    # fitz.open() reads the PDF file into memory and gives us a document object
    # we can iterate over page by page.
    # 'with' is a context manager — it automatically closes the file when done,
    # even if an error occurs. Always use 'with' when opening files.
    with fitz.open(path) as pdf:
        # 'enumerate' gives us both the index (page_num) and the value (page)
        # as we loop — so page_num starts at 0, we add 1 to make it human-readable
        for page_num, page in enumerate(pdf, start=1):
            # get_text() extracts all the text from this page as a single string.
            # "text" mode reads in natural reading order (top to bottom, left to right).
            # strip() removes leading/trailing whitespace and newlines.
            text = page.get_text("text").strip()

            # Skip pages with no meaningful text.
            # Some pages are just images, decorative dividers, or blank —
            # adding empty Documents would just pollute our vector store with noise.
            if not text:
                continue

            # Build a LangChain Document for this page.
            # page_content: the actual text the LLM will read
            # metadata: information about WHERE this text came from
            #           this is what powers our source citations later
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(path),  # full file path
                    "file_name": path.name,  # just "report.pdf"
                    "page": page_num,  # page number (1-indexed)
                    "total_pages": len(pdf),  # total pages in the document
                },
            )
            documents.append(doc)

    return documents


def _load_with_pdfplumber(path: Path) -> list[Document]:
    """
    Extract text from a PDF using pdfplumber.
    Used as a fallback when PyMuPDF extracts very little text.
    pdfplumber is slower but handles complex layouts and tables better.
    """
    documents = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # extract_text() is pdfplumber's equivalent of fitz's get_text()
            # It returns None if a page has no text (e.g. a full-image page)
            # so we use 'or ""' to convert None into an empty string safely
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "page": page_num,
                    "total_pages": len(pdf.pages),
                },
            )
            documents.append(doc)

    return documents


def load_pdfs_from_folder(folder_path: str | Path) -> list[Document]:
    """
    Load all PDF files from a folder.
    Returns a single flat list of Documents from all PDFs combined.

    This is useful when a user uploads multiple documents at once —
    we process them all and store them together in ChromaDB.
    """
    folder = Path(folder_path)

    # glob("*.pdf") finds all files ending in .pdf in this folder.
    # We sort them so processing order is consistent and predictable.
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"  ⚠️  No PDF files found in: {folder}")
        return []

    all_documents = []
    for pdf_file in pdf_files:
        print(f"\n📄 Loading: {pdf_file.name}")
        docs = load_pdf(pdf_file)
        all_documents.extend(docs)  # extend adds all items, not just the list itself

    print(f"\n📚 Total: {len(all_documents)} pages loaded from {len(pdf_files)} PDFs")
    return all_documents
