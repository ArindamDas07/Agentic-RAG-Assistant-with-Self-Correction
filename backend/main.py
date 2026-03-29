from fastapi import FastAPI, UploadFile, File, HTTPException,Form
from typing import List,Annotated
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uuid
from backend.graph_details import build_graph, format_output
import os
from backend.pdf_utils import get_pdf_text,get_text_chunks,create_vector_store
from pathlib import Path
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Upload multiple PDF files (max 5) and return a unique user_id.
    Validates file type and size (<=10MB per file).
    """
    # Limit check
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")

    user_id = str(uuid.uuid4())
    user_folder = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_folder, exist_ok=True)

    for file in files:
        # Validate extension
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        
        # Read content to check size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds 10MB")
        
        # Save file
        file_path = os.path.join(user_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        
        await file.seek(0)

    return {"user_id": user_id}

@app.get("/uploads/{user_id}")
async def list_uploaded(user_id: str):
    """
    List all uploaded PDF filenames for a given user_id.
    Returns an empty list if no files exist.
    """
    user_folder = os.path.join(UPLOAD_DIR, user_id)
    if not os.path.exists(user_folder):
        return JSONResponse({"files": []}) 
    files = os.listdir(user_folder)
    return JSONResponse({"files": files})

# Delete a PDF (pre-embedding only)
@app.delete("/uploads/{user_id}/{filename}")
async def delete_(user_id:str,filename:str):
    """
    Delete a specific uploaded PDF for a given user.
    Raises 404 if the file does not exist.
    """
    user_folder = os.path.join(UPLOAD_DIR,user_id)
    file_path = os.path.join(user_folder,filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404,detail=f"{filename} not found for the user")
    os.remove(file_path)
    return {"details": f"{filename} deleted successfully"}

@app.post("/process-pdf/{user_id}")
async def process_pdfs(user_id:str):
    """
    Process all PDFs uploaded by a user:
    - Extracts text from PDFs
    - Splits text into chunks
    - Creates/updates the per-user Chroma vector store
    
    Returns the number of PDFs processed and text chunks created.
    """
    user_folder = Path(UPLOAD_DIR)/user_id
    if not user_folder.exists():
        raise HTTPException(status_code=404, detail="User Not Found")
    
    pdf_files = list(user_folder.glob('*.pdf'))
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF to Process")
    
    #extract text
    docs = get_pdf_text(pdf_files)

    #chunk text
    chunks = get_text_chunks(docs)

    # Use per-user collection to avoid overwriting
    vector_store = create_vector_store(chunks, persist_dir="chroma_db", user_id=user_id)

    return {"details":f"{len(pdf_files)} PDFs processed", "chunks": len(chunks)}

@app.post("/ask/{user_id}")
async def ask_question(user_id:str, question:str =Form(...)):
    """
    Ask a question for a specific user_id.
    The user_id corresponds to the folder in uploads and the Chroma DB.
    """
    #check if the user*s vector exits
    user_db_path = Path(f"chroma_db/{user_id}")
    if not user_db_path.exists():
        raise HTTPException(status_code=404,detail=f"No processed PDFs found for user_id {user_id}")
    try:
        graph = build_graph()
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Failed to build QA chain: {e}")    
    #invoke the graph with question
    state = graph.invoke({"question": question,"user_id":user_id})
    # Format output for user
    formatted = format_output(state)
    return {"answer": formatted}
# 🔥 ALWAYS KEEP THIS AT THE VERY END
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")