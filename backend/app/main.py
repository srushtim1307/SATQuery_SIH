from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings, PREVIEWS_DIR, UPLOADS_DIR, PROCESSED_DIR
from app.api.routes_health import router as health_router
from app.api.routes_upload import router as upload_router
from app.api.routes_validation import router as validation_router
from app.api.routes_analysis import router as analysis_router
from app.db.database import Base, engine

# Ensure all database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis (ISRO SIH26167)"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directories for uploads and processed outputs/previews
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")
app.mount("/static/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")

# Include Routers
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(validation_router)
app.include_router(analysis_router)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.app_name} API ({settings.version})",
        "docs": "/docs",
        "health": "/api/health",
        "uploads": "/api/uploads",
        "validate": "/api/validate",
        "analyze": "/api/analyze",
        "models": "/api/models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
