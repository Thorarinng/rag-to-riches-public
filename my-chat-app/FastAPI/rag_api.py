# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from rag.task3_rag_completed_solution import full_rag_pipeline  # import your RAG function

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/rag", response_model=QueryResponse)
def rag_endpoint(req: QueryRequest):
    answer = full_rag_pipeline(req.query)
    return {"answer": answer}


@app.post("/rag-stream")
def rag_stream(req: QueryRequest):
    def generate():
        # Wrap your pipeline to yield chunks
        # Example: assume full_rag_pipeline can return an iterable of strings
        for chunk in full_rag_pipeline(req.query):  
            yield chunk
    return StreamingResponse(generate(), media_type="text/plain")
