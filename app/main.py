"""FastAPI application with minimal web UI."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import OUTPUT_DIR, UPLOAD_DIR
from app.llm_client import check_model_ready
from app.mapping_store import MappingStore
from app.orchestrator import process_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pseudonymisierungs-Workflow MVP", version="0.1.0")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".md", ".txt"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    model_ready = await check_model_ready()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_ready": model_ready,
    })


@app.get("/health")
async def health():
    model_ready = await check_model_ready()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/process")
async def process(
    request: Request,
    file: UploadFile = File(...),
    project_id: str = Form(default="default"),
):
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"},
        )

    data = await file.read()
    file_size = len(data)

    # Save original upload
    upload_path = os.path.join(UPLOAD_DIR, filename)
    with open(upload_path, "wb") as f:
        f.write(data)

    # Process
    result = await process_document(data, filename, project_id)

    # Save output
    output_filename = f"pseudonymized_{Path(filename).stem}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.final_text)

    # Save mapping as JSON
    mapping_filename = f"mapping_{project_id}.json"
    mapping_path = os.path.join(OUTPUT_DIR, mapping_filename)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(result.mappings, f, ensure_ascii=False, indent=2)

    # Build template context
    steps_data = []
    for step in result.steps:
        steps_data.append({
            "name": step.name,
            "label": step.label,
            "status": step.status.value,
            "duration": step.duration_seconds,
            "detail": step.detail,
        })

    template_result = {
        "status": result.status.value,
        "filename": filename,
        "file_size": file_size,
        "project_id": project_id,
        "output_filename": output_filename,
        "mapping_filename": mapping_filename,
        "entities_found": result.entities_found,
        "entities_pseudonymized": result.entities_pseudonymized,
        "processing_time": result.processing_time_seconds,
        "warnings": result.warnings,
        "steps": steps_data,
        # Texts for each stage
        "extracted_text": result.extracted_text[:10000],
        "extracted_blocks": result.extracted_blocks,
        "entities": result.entities,
        "pseudonymized_text": result.pre_generalization_text[:10000],
        "final_text": result.final_text[:10000],
        "mappings": result.mappings,
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "model_ready": True,
            "result": template_result,
        })

    return {
        **template_result,
        "extracted_text": result.extracted_text,
        "pseudonymized_text": result.pre_generalization_text,
        "final_text": result.final_text,
    }


@app.get("/download/{artifact_type}/{filename}")
async def download(artifact_type: str, filename: str):
    if artifact_type == "output":
        path = os.path.join(OUTPUT_DIR, filename)
    elif artifact_type == "mapping":
        path = os.path.join(OUTPUT_DIR, filename)
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid artifact type"})

    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "File not found"})

    def iterfile():
        with open(path, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/mappings/{project_id}")
async def get_mappings(project_id: str):
    store = MappingStore(project_id)
    mappings = store.get_all_mappings()
    store.close()
    return {"project_id": project_id, "mappings": mappings}


@app.get("/mappings/{project_id}/csv")
async def get_mappings_csv(project_id: str):
    store = MappingStore(project_id)
    mappings = store.get_all_mappings()
    store.close()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["category", "original", "pseudonym", "context"])
    writer.writeheader()
    for m in mappings:
        writer.writerow(m)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=mapping_{project_id}.csv"},
    )
