from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import uvicorn
import logging
from datetime import datetime
import os
from pathlib import Path

from pdf_processor import PDFSearchEngine
from search_result import SearchResult

# API-Modelle
class SearchQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Die Suchanfrage")
    top_k: int = Field(default=3, ge=1, le=100, description="Anzahl der gewünschten Ergebnisse")
    min_score: float = Field(default=0.3, ge=0, le=1, description="Minimaler Ähnlichkeitsscore")
    document_filter: Optional[str] = Field(None, description="Optional: Nur in diesem Dokument suchen")

class SearchResponse(BaseModel):
    query: str
    timestamp: datetime
    total_results: int
    results: List[Dict]
    execution_time_ms: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# API-Konfiguration
class APIConfig:
    def __init__(self):
        self.upload_dir = Path("uploaded_pdfs")
        self.upload_dir.mkdir(exist_ok=True)
        self.api_keys = os.environ.get("API_KEYS", "test_key").split(",")
        self.max_file_size = 100 * 1024 * 1024  # 100MB

# API Setup
app = FastAPI(
    title="PDF Semantic Search API",
    description="API für semantische Suche in PDF-Dokumenten",
    version="1.0.0"
)

api_key_header = APIKeyHeader(name="X-API-Key")
config = APIConfig()
search_engine = PDFSearchEngine(persist_directory="./chroma_db")
logger = logging.getLogger(__name__)

# Authentifizierung
async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key not in config.api_keys:
        raise HTTPException(
            status_code=403,
            detail="Ungültiger API-Schlüssel"
        )
    return api_key

# Endpunkte
@app.post("/documents/upload", response_model=Dict)
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """PDF-Dokument hochladen und verarbeiten"""
    try:
        # Validierung
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Nur PDF-Dateien sind erlaubt")
        
        file_path = config.upload_dir / file.filename
        
        # Datei speichern
        try:
            content = await file.read()
            if len(content) > config.max_file_size:
                raise HTTPException(400, "Datei zu groß (max. 100MB)")
                
            with open(file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(500, f"Fehler beim Speichern der Datei: {str(e)}")
        
        # PDF verarbeiten
        success, error_message = search_engine.load_pdf(str(file_path))
        
        if not success:
            raise HTTPException(400, error_message or "Fehler bei der PDF-Verarbeitung")
        
        return {
            "message": "Dokument erfolgreich verarbeitet",
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler beim Dokumenten-Upload: {str(e)}")
        raise HTTPException(500, "Interner Serverfehler")

@app.post("/search", response_model=SearchResponse)
async def search_documents(
    query: SearchQuery,
    api_key: str = Depends(verify_api_key)
):
    """Semantische Suche in den Dokumenten"""
    start_time = datetime.now()
    
    try:
        # Filter vorbereiten
        filter_dict = {"document": query.document_filter} if query.document_filter else None
        
        # Suche durchführen
        results = search_engine.search(
            query=query.query,
            top_k=query.top_k,
            min_score=query.min_score,
            filter_dict=filter_dict,
            format_output=False
        )
        
        # Fehlerbehandlung
        if isinstance(results, tuple) and not results[0]:
            raise HTTPException(400, results[1])
        
        # Antwort formatieren
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SearchResponse(
            query=query.query,
            timestamp=datetime.now(),
            total_results=len(results),
            results=[r.to_dict() for r in results],
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fehler bei der Suche: {str(e)}")
        raise HTTPException(500, "Interner Serverfehler")

@app.get("/documents", response_model=List[str])
async def list_documents(
    api_key: str = Depends(verify_api_key)
):
    """Liste aller verfügbaren Dokumente"""
    try:
        documents = search_engine.vector_store.list_documents()
        return documents
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Dokumentenliste: {str(e)}")
        raise HTTPException(500, "Interner Serverfehler")

# Error Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unbehandelter Fehler: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Interner Serverfehler",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

# Server starten
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 