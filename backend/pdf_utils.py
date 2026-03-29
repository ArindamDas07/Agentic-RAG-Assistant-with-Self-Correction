from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


# ✅ LOAD ONCE
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#extract pdf text
def get_pdf_text(pdf_paths:list[str]):
    """
    Extract text from a list of PDF files and return as LangChain Document objects.

    This function takes one or more PDF file paths, loads each PDF using PyPDFLoader,
    and converts the content into a list of LangChain Document objects, which can
    then be used for downstream processing such as embeddings or retrieval.

    Parameters:
        pdf_paths (list[str]): A list of file paths pointing to PDF files.

    Returns:
        list[Document]: A list of LangChain Document objects containing the text from
        all PDFs in the order they were provided.
    """
    docs = []
    for pdf_path in pdf_paths:
        pdf_path = str(pdf_path)
        loader = PyPDFLoader(pdf_path)
        loader_docs = loader.load()
        docs.extend(loader_docs)
    return docs

#split document into chunks
def get_text_chunks(docs, chunk_size=900, chunk_overlap = 150):
    """
    Split LangChain Document objects into smaller text chunks for processing.

    This function takes a list of LangChain Document objects and splits their content
    into smaller overlapping chunks using RecursiveCharacterTextSplitter. These chunks
    are suitable for vectorization, embeddings, or retrieval tasks where long documents
    need to be broken into manageable pieces.

    Parameters:
        docs (list[Document]): A list of LangChain Document objects to be split.
        chunk_size (int, optional): Maximum number of characters per chunk. Default is 900.
        chunk_overlap (int, optional): Number of characters that overlap between consecutive chunks. Default is 150.

    Returns:
        list[Document]: A list of LangChain Document objects representing the chunked text.
    """
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )    
    chunks = text_spliter.split_documents(docs)
    return chunks

def create_vector_store(chunks, user_id="default", persist_dir="chroma_db"):
    """
    Create a per-user Chroma vector store from text chunks.

    This function initializes a Chroma vector database for a specific user, 
    adds the provided text chunks as documents, and persists the embeddings 
    on disk. Each user has a separate folder under the specified `persist_dir`.

    Parameters:
        chunks (list[Document]): List of LangChain Document objects (text chunks) to be embedded.
        user_id (str, optional): Unique identifier for the user. Default is "default".
        persist_dir (str, optional): Base directory where user vector stores are saved. Default is "chroma_db".

    Returns:
        Chroma: The initialized and populated Chroma vector store for the user.
    """
    
    # ✅ Folder per user
    persist_path = Path(persist_dir) / user_id
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name="chroma_collection",  # same collection name for all users
        embedding_function=embedding_model,
        persist_directory=str(persist_path)    # embeddings will be stored here
    )
    vector_store.add_documents(chunks)
    return vector_store
