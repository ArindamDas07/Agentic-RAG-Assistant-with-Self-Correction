from fastapi import FastAPI, UploadFile, File, HTTPException,Form,BackgroundTasks
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import aiofiles
import asyncio
import logging
import uuid
from backend.graph_details import build_graph, format_output
import traceback
from backend.pdf_utils import get_pdf_text,get_text_chunks,create_vector_store
from pathlib import Path

app = FastAPI()



# =========================
# Logging 
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Directories
# =========================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# =========================
# Graph 
# =========================
try:
    GRAPH = build_graph()
except Exception as e:
    raise HTTPException(status_code=500,detail=f"Failed to build QA chain: {e}")

# =========================
# Processing status
# =========================
processing_status = {}


# =========================
# Upload PDFs 
# =========================
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
    user_folder = UPLOAD_DIR / user_id
    user_folder.mkdir(parents=True, exist_ok=True)

    for file in files:
        # Validate extension
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        
        # Read content to check size
        # Reads the entire content of the uploaded file asynchronously.
        # await ensures the server can handle other requests while waiting.
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds 10MB")
        
        # Save file
        #Path(file.filename).name extracts a safe filename, stripping any directory paths to avoid security issues.
        safe_name = Path(file.filename).name 
        file_path = user_folder/safe_name
        
        # Opens the file asynchronously for writing in binary mode (wb) using aiofiles.
        # Writes the file content to disk without blocking the event loop.
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        # Resets the file’s internal pointer to the start.
        # Useful if you want to read the file again later (for example, for processing).
        await file.seek(0)
    logger.info(f"User {user_id} uploaded {len(files)} files")
    return {"user_id": user_id}

# =========================
# List uploaded PDFs
# =========================
@app.get("/uploads/{user_id}")
async def list_uploaded(user_id: str):
    """
    List all uploaded PDF filenames for a given user_id.
    Returns an empty list if no files exist.
    """
    user_folder = UPLOAD_DIR / user_id
    if not user_folder.exists():
        return JSONResponse({"files": []}) 
    files = [f.name for f in user_folder.glob("*.pdf")]
    return JSONResponse({"files": files})

# =========================
# Delete PDF
# =========================
@app.delete("/uploads/{user_id}/{filename}")
async def delete_(user_id:str,filename:str):
    """
    Delete a specific uploaded PDF for a given user.
    Raises 404 if the file does not exist.
    """
    user_folder = UPLOAD_DIR / user_id
    # Path(filename).name ensures a safe filename by stripping any directory traversal or extra path segments.
    # Combines it with user_folder to get the full path to the file.
    file_path = user_folder/Path(filename).name

    if not file_path.exists():
        raise HTTPException(status_code=404,detail=f"{filename} not found for the user")
    
    # Deletes the file from the filesystem.
    #unlink() is the Pathlib method for removing files.
    file_path.unlink()

    logger.info(f"Deleted {filename} for user {user_id}")
    return {"details": f"{filename} deleted successfully"}

# =========================
# PDF Processing 
# =========================
async def run_processing(user_id:str):
    processing_status[user_id] = "running"
    try:
        user_folder = UPLOAD_DIR/user_id

        if not user_folder.exists():
            processing_status[user_id] = "error: user not found"
            return
        
        pdf_files = list(user_folder.glob("*.pdf"))
        if not pdf_files:
            processing_status[user_id] = "error: no pdfs found"
            return
        # Runs the function in a separate thread
        # Prevents blocking the main event loop
        docs = await asyncio.to_thread(get_pdf_text,pdf_files)
        chunks = await asyncio.to_thread(get_text_chunks,docs)
        await asyncio.to_thread(create_vector_store,chunks, persist_dir="chroma_db", user_id=user_id)
        processing_status[user_id] = "completed"
        logger.info(f"Processed PDFs for the user {user_id}")
    except Exception as e:
        processing_status[user_id] = f"error: {e}"
        logger.error(f"Processing failed for {user_id}: {traceback.format_exc()} ")
          

@app.post("/process-pdf/{user_id}")
async def process_pdfs(user_id:str, background_tasks:BackgroundTasks):
    """
    Process all PDFs uploaded by a user:
    - Extracts text from PDFs
    - Splits text into chunks
    - Creates/updates the per-user Chroma vector store
    
    Returns the number of PDFs processed and text chunks created.
    """
    background_tasks.add_task(run_processing,user_id)

    return {"status":"started"}

# =========================
# Processing status API
# =========================
@app.get("/status/{user_id}")
async def get_status(user_id:str):
    return {"status": processing_status.get(user_id,"not_started")}

# =========================
# Ask question 
# =========================
@app.post("/ask/{user_id}")
async def ask_question(user_id:str, question:str =Form(...)):
    """
    Ask a question for a specific user_id.
    The user_id corresponds to the folder in uploads and the Chroma DB.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    #check if the user*s vector exits
    user_db_path = Path(f"chroma_db/{user_id}")
    if not user_db_path.exists():
        raise HTTPException(status_code=404,detail=f"No processed PDFs found for user_id {user_id}")
    
    try:
        # Invoke graph
        state = await asyncio.to_thread(GRAPH.invoke, {"question": question, "user_id": user_id})
        
        # Format output (with emojis ✅)
        formatted = format_output(state)

        return {"answer": formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 🔥 ALWAYS KEEP THIS AT THE VERY END
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")