from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from classifier import ZooLensClassifier
import uvicorn
import logging
import time

import os
from pathlib import Path

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ZooLens AI")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = BASE_DIR / "static"
if not static_dir.exists():
    logger.error(f"Static directory not found at {static_dir}")
    # Create it to prevent immediate crash, though it will be empty
    static_dir.mkdir()

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = BASE_DIR / "templates"
if not templates_dir.exists():
    logger.error(f"Templates directory not found at {templates_dir}")
    templates_dir.mkdir()
    
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize Classifier
logger.info("Initializing Classifier...")
classifier = ZooLensClassifier()
logger.info("Classifier Initialized.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Home page accessed.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_species(file: UploadFile = File(...)):
    start_time = time.time()
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        logger.info(f"Received image: {file.filename} ({len(contents)} bytes)")
        
        predictions = classifier.predict(contents)
        
        duration = time.time() - start_time
        logger.info(f"Prediction completed in {duration:.2f}s")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
