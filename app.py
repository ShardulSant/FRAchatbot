from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi.responses import FileResponse
import os
import shutil
import uuid
app = FastAPI(title="TinyLlama RAG API - Separate Endpoints")

# Global state
CHROMA_PATH = "./chroma_db"
embeddings = None
vectorstore = None
llm = None

class ChatRequest(BaseModel):
    message: str

class PDFResponse(BaseModel):
    status: str
    chunks_added: int

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstore, llm
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    llm = ChatOllama(model="tinyllama")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

def store_in_bucket(temp_path: str, original_filename: str):
    os.makedirs("./uploads", exist_ok=True)

    document_id = str(uuid.uuid4())
    final_path = f"./uploads/{document_id}.pdf"

    # Move temp file → uploads
    shutil.copy(temp_path, final_path)

    return {
        "document_id": document_id,
        "pdf_path": final_path,
        "original_filename": original_filename,
        "file_size": os.path.getsize(final_path)
    }

@app.post("/upload-pdf", response_model=PDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        pdf_path= f"./temp_{file.filename}"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        bucket_meta  = store_in_bucket(pdf_path,file.filename)       
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "pdf_path": bucket_meta["pdf_path"],
                "source": bucket_meta["original_filename"],
                "file_size": bucket_meta["file_size"]
            })
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            if 'page' in chunk.metadata:
                chunk.metadata['page'] = chunk.metadata['page'] + 1
    
        vectorstore.add_documents(chunks)
        print("\n=== CHUNK PAGE NUMBERS ===")
        for idx, chunk in enumerate(chunks):
            page_num = chunk.metadata.get('page', 'N/A')
            source = chunk.metadata.get('source', 'N/A')
            print(f"Chunk {idx + 1}: Page {page_num} from {source}")
            print(chunk.metadata)
        print(f"Total chunks created: {len(chunks)}\n")
        os.remove(pdf_path)
        
        return PDFResponse(
            status="success", 
            chunks_added=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="No documents indexed. Upload PDF first.")
    
    try:
        relevant_docs = vectorstore.similarity_search(request.message, k=4)
        print(relevant_docs)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt_template = """Use ONLY the following context to answer the question. 
            If the answer isn't in the context, say "I don't have that information".

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        formatted_prompt = prompt_template.format(
            context=context,
            question=request.message
        )
        
        response = llm.invoke(formatted_prompt)
        pdf_page_list = []

        for doc in relevant_docs:
            pdf_path = doc.metadata.get("pdf_path")       # path to PDF
            page_number = doc.metadata.get("page")  # or "page" if you stored numeric page
            pdf_page_list.append([pdf_path, page_number])
        print(pdf_page_list)        
        return {
            "response": response.content,
            "context_chunks": len(relevant_docs),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-db/")
async def clear_db():
    global vectorstore
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    vectorstore = None
    return {"status": "Database cleared"}

@app.get("/status")
async def status():
    """Check system status"""
    has_docs = vectorstore is not None and os.path.exists(CHROMA_PATH)
    return {
        "chroma_ready": has_docs,
        "chroma_path": CHROMA_PATH,
        "llm_loaded": llm is not None
    }

## === HTML UI ===
@app.get("/", response_class=HTMLResponse)
async def html_ui():
    return "Hello World"

@app.get("/pdf/{doc_id}")
def open_pdf(doc_id: str):
    pdf_path = f"docs/{doc_id}.pdf"

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{doc_id}.pdf",
        headers={
            "Content-Disposition": "inline"
        }
    )

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)