from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

from pathlib import Path
import os
from typing import List
import shutil


logger = logging.getLogger(__name__)
# =========================
# ✅ LOAD EMBEDDING MODEL ONCE
# =========================
embedding_model = HuggingFaceEmbeddings(
    model=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
)

#extract pdf text
def get_pdf_text(pdf_paths:List[Path]):
    """
    Extract text from multiple PDFs safely.
    Skips corrupted PDFs instead of crashing.
    """
    docs = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs_from_file = loader.load()
            # add metadata
            for doc in docs_from_file:
                doc.metadata['source'] = pdf_path.name

            docs.extend(docs_from_file)    
        except Exception:
            logger.exception(f"Failed to load PDF {pdf_path}")    
    if not docs:
        logger.error("No text could be extracted from any PDFs")    
        raise ValueError("No text could be extracted from PDFs")    
                
    logger.info(f"Loaded {len(docs)} documents")
    return docs

#split document into chunks
def get_text_chunks(docs, chunk_size=900, chunk_overlap = 150):
    """
    Split documents into overlapping chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )    
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        logger.error("No chunks created from documents")
        raise ValueError("No valid chunks created from documents")
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks, user_id="default", persist_dir="chroma_db"):
    """
    Create (or overwrite) a per-user Chroma vector store.

    IMPORTANT:
    - Deletes old DB to prevent duplicate embeddings
    - Persists embeddings to disk
    """
    
    # ✅ Folder per user
    persist_path = Path(persist_dir) / user_id

    #remove old DB to prevent duplication problem
    if persist_path.exists():
        shutil.rmtree(persist_path) #shutil.rmtree(...) deletes the entire directory recursively

    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name="chroma_collection",  # same collection name for all users
        embedding_function=embedding_model,
        persist_directory=str(persist_path)    # embeddings will be stored here
    )
    vector_store.add_documents(chunks)
    
    logger.info(f"Vector DB created at {persist_path}")
    
