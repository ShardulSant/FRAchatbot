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
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

import os
import shutil
import uuid
app = FastAPI(title="TinyLlama RAG API - Separate Endpoints")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],        # Authorization, Content-Type, etc.
)
# Global state
CHROMA_PATH = "./chroma_db"
embeddings = None
vectorstore = None
llm = None

class HistoryRequest(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[HistoryRequest] = []
    language:str

class PDFResponse(BaseModel):
    status: str
    chunks_added: int

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstore, llm
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    llm = ChatOllama(model="llama3.2")
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
        history_text = "\n".join(f"{h.role}: {h.content}" for h in request.history)
        prompt_template = """You are a helpful and accurate assistant.
        Primary Rule:
        - Use the provided context ONLY if it is clearly relevant to the user's question.
        - If the context is irrelevant, partially relevant, or does not contain the required information, IGNORE it completely.

        Context Usage Rules:
        1. If the answer can be fully derived from the context, use it.
        2. If the context does not contain the answer, respond using general, well-known, and correct information.
        3. Never force an answer from the context if it does not logically apply to the question.
        4. Do not mention the context explicitly in the final answer.

        Conversation History Rules:
        - Use conversation history only to understand intent or resolve ambiguity.
        - Do not reuse outdated or unrelated information from history.

        Answer Generation Rules:
        - Answer strictly based on the user’s question.
        - Ensure factual correctness and clarity.
        - Do not hallucinate missing facts.
        - Keep the answer concise but complete.
        - Do not use emojis.
        - Respond in the specified language using its native script.
        LANGUAGE CONSTRAINT (MANDATORY):
        - You MUST respond ONLY in {language}.
        - Use ONLY native script of {language}.
        - Do NOT mix languages.
        - Do NOT translate names, code, or technical terms unless required.
        Inputs:
        Context:
        {context}

        Conversation History:
        {history}

        Question:
        {question}

        Output Language:
        {language}

        Final Instruction:
        Generate the best possible answer for the user, following all rules above.

        Answer:
        """
        
        formatted_prompt = prompt_template.format(
            context=context,
            question=request.message,
            history = "" ,
            language = request.language
        )# Add history = history_text
        print(request.language)
        response = llm.invoke(formatted_prompt)
        answer_text = response.content.strip()

        # 🔹 Grounding check
        in_context = not answer_text.lower().startswith("i don't have")
        pdf_page_list = []

        pdf_page_list = [
            {"pdf_path": doc.metadata.get("pdf_path"), "page": doc.metadata.get("page")}
            for doc in relevant_docs
        ]

        print(pdf_page_list)        
        return {
            "response": response.content,
            "context_chunks": len(relevant_docs),
            "pdf_page_list":pdf_page_list,
            "inContext":in_context
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

@app.get("/", response_class=HTMLResponse)
async def html_ui():
    return "Hello World"

@app.get("/open-pdf")
def open_pdf(
    pdf_path: str = Query(...),
):
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline"
        }
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)