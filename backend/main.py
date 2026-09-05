"""
Resume Reader & RAG Engine - FastAPI Server
Exposes REST APIs for PDF extraction, hybrid vector indexing, RAG Q&A, and resume intelligence tools.
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .pdf_parser import parse_document
from .rag_engine import ResumeRAGStore
from .resume_analyzer import (
    extract_profile,
    match_job_description,
    critique_resume,
    generate_interview_questions
)
from .sample_data import SAMPLE_RESUMES

app = FastAPI(
    title="Resume RAG Intelligence Engine",
    description="End-to-End Retrieval-Augmented Generation (RAG) System for Resumes",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Store
rag_store = ResumeRAGStore()
active_resume_id: Optional[str] = None

# Pre-load sample resumes on startup
for sample_id, sample_doc in SAMPLE_RESUMES.items():
    rag_store.add_resume(sample_id, sample_doc)
active_resume_id = "alex_chen_senior_ai_engineer"

# Static directory path
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# Pydantic Request Models
class QueryRequest(BaseModel):
    query: str
    resume_id: Optional[str] = None
    top_k: Optional[int] = 4

class JobMatchRequest(BaseModel):
    job_description: str
    resume_id: Optional[str] = None

class ApiKeyRequest(BaseModel):
    api_key: str

class SelectResumeRequest(BaseModel):
    resume_id: str

class LoadSampleRequest(BaseModel):
    sample_id: str

@app.get("/api/status")
def get_status():
    global active_resume_id
    return {
        "status": "online",
        "gemini_active": rag_store.gemini_available,
        "minilm_active": bool(getattr(rag_store.minilm, "available", False)),
        "embedding_backend": (
            "minilm" if getattr(rag_store.minilm, "available", False) else "tfidf"
        ),
        "active_resume_id": active_resume_id,
        "total_resumes_loaded": len(rag_store.resumes),
        "resumes": [
            {
                "id": rid,
                "title": doc.get("title") or doc.get("filename", "Resume"),
                "filename": doc.get("filename", "resume"),
                "total_pages": doc.get("total_pages", 1),
                "total_chunks": len(rag_store.chunks.get(rid, []))
            }
            for rid, doc in rag_store.resumes.items()
        ]
    }

@app.post("/api/api-key")
def set_api_key(req: ApiKeyRequest):
    success, msg = rag_store.set_api_key(req.api_key)
    return {
        "success": success,
        "message": msg,
        "gemini_active": rag_store.gemini_available
    }

@app.post("/api/upload")
async def upload_resume(file: UploadFile = File(...)):
    global active_resume_id
    try:
        content = await file.read()
        filename = file.filename or "uploaded_resume.pdf"
        
        # Parse document
        doc_data = parse_document(content, filename)
        resume_id = f"res_{uuid.uuid4().hex[:8]}"
        doc_data["id"] = resume_id
        doc_data["title"] = filename
        
        # Add to RAG Store
        index_result = rag_store.add_resume(resume_id, doc_data)
        active_resume_id = resume_id

        return {
            "success": True,
            "resume_id": resume_id,
            "filename": filename,
            "total_pages": doc_data.get("total_pages", 1),
            "total_characters": doc_data.get("total_characters", 0),
            "total_chunks": index_result["total_chunks"],
            "sections_found": index_result["sections_found"],
            "embedding_mode": index_result["embedding_mode"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/load-sample")
def load_sample(req: LoadSampleRequest):
    global active_resume_id
    sample_id = req.sample_id
    if sample_id not in SAMPLE_RESUMES:
        raise HTTPException(status_code=404, detail="Sample resume not found.")
    
    sample_doc = SAMPLE_RESUMES[sample_id]
    rag_store.add_resume(sample_id, sample_doc)
    active_resume_id = sample_id

    return {
        "success": True,
        "resume_id": sample_id,
        "title": sample_doc["title"],
        "filename": sample_doc["filename"],
        "total_chunks": len(rag_store.chunks.get(sample_id, []))
    }

@app.post("/api/select-resume")
def select_resume(req: SelectResumeRequest):
    global active_resume_id
    if req.resume_id not in rag_store.resumes:
        raise HTTPException(status_code=404, detail="Resume ID not found.")
    active_resume_id = req.resume_id
    return {"success": True, "active_resume_id": active_resume_id}

@app.post("/api/query")
def rag_query(req: QueryRequest):
    global active_resume_id
    resume_id = req.resume_id or active_resume_id
    if not resume_id or resume_id not in rag_store.resumes:
        raise HTTPException(status_code=400, detail="No active resume selected. Please upload a resume first.")

    # 1. Retrieve top-k chunks
    retrieved = rag_store.retrieve(resume_id, req.query, top_k=req.top_k or 4)
    
    # 2. Generate grounded answer
    gen_result = rag_store.generate_answer(resume_id, req.query, retrieved)

    # 3. Dynamic follow-up questions
    suggested_followups = [
        "What technical skills are listed on this resume?",
        "Summarize the projects on this file.",
        "What education and internships are mentioned?",
        "Give me a short overview of this candidate."
    ]

    return {
        "query": req.query,
        "resume_id": resume_id,
        "answer": gen_result["answer"],
        "model_used": gen_result["model_used"],
        "mode": gen_result["mode"],
        "sources": gen_result["sources"],
        "suggested_questions": suggested_followups
    }

@app.get("/api/chunks")
def get_chunks(resume_id: Optional[str] = None):
    global active_resume_id
    rid = resume_id or active_resume_id
    if not rid or rid not in rag_store.chunks:
        return {"chunks": []}
    
    chunks_list = [c.to_dict() for c in rag_store.chunks[rid]]
    return {
        "resume_id": rid,
        "total_chunks": len(chunks_list),
        "chunks": chunks_list
    }

@app.get("/api/analyze/profile")
def get_profile(resume_id: Optional[str] = None):
    global active_resume_id
    rid = resume_id or active_resume_id
    if not rid or rid not in rag_store.resumes:
        raise HTTPException(status_code=400, detail="No active resume selected.")
    
    doc = rag_store.resumes[rid]
    profile = extract_profile(doc)
    return profile

@app.post("/api/analyze/job-match")
def evaluate_job_match(req: JobMatchRequest):
    global active_resume_id
    rid = req.resume_id or active_resume_id
    if not rid or rid not in rag_store.resumes:
        raise HTTPException(status_code=400, detail="No active resume selected.")
    
    doc = rag_store.resumes[rid]
    result = match_job_description(doc, req.job_description, rag_store.api_key)
    return result

@app.get("/api/analyze/critique")
def get_critique(resume_id: Optional[str] = None):
    global active_resume_id
    rid = resume_id or active_resume_id
    if not rid or rid not in rag_store.resumes:
        raise HTTPException(status_code=400, detail="No active resume selected.")
    
    doc = rag_store.resumes[rid]
    result = critique_resume(doc, rag_store.api_key)
    return result

@app.get("/api/analyze/interview")
def get_interview_questions(resume_id: Optional[str] = None):
    global active_resume_id
    rid = resume_id or active_resume_id
    if not rid or rid not in rag_store.resumes:
        raise HTTPException(status_code=400, detail="No active resume selected.")
    
    doc = rag_store.resumes[rid]
    questions = generate_interview_questions(doc, rag_store.api_key)
    return {"questions": questions}

# Mount static frontend
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
