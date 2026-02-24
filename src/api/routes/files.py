from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.dependencies import get_request_id
from src.core.file_manager import list_project_files
from src.core.logger import get_logger
from src.core.security import ensure_project_path
from src.graph.runner import get_experiment_or_404
from src.schemas.response_schemas import error_payload, response_envelope

router = APIRouter()
logger = get_logger(__name__)


@router.get("/research/{experiment_id}/files")
async def get_files(experiment_id: str, request_id: str = Depends(get_request_id)):
    logger.info("api.files.list", request_id=request_id, experiment_id=experiment_id)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    project = Path(state["project_path"]).resolve()
    files = []
    total_size = 0
    for path in list_project_files(state["project_path"]):
        rel = str(path.relative_to(project))
        stat = path.stat()
        total_size += stat.st_size
        files.append(
            {
                "path": rel,
                "absolute_path": str(path),
                "size_bytes": stat.st_size,
                "created_at": stat.st_ctime,
                "phase": state["phase"],
                "is_quantum": rel.endswith("quantum_circuit.py"),
            }
        )

    data = {
        "experiment_id": experiment_id,
        "project_path": state["project_path"],
        "files": files,
        "total_files": len(files),
        "total_size_bytes": total_size,
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}/files/{file_path:path}")
async def get_file_content(experiment_id: str, file_path: str, request_id: str = Depends(get_request_id)):
    logger.info("api.files.get", request_id=request_id, experiment_id=experiment_id, file_path=file_path)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=error_payload("EXPERIMENT_NOT_FOUND", f"Experiment {experiment_id} does not exist"))

    decoded = unquote(file_path)
    absolute = str(Path(state["project_path"]) / decoded)
    try:
        target = ensure_project_path(absolute, state["project_path"])
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=error_payload("FORBIDDEN_PATH", str(exc)))
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=error_payload("FILE_NOT_FOUND", decoded))

    data = {
        "experiment_id": experiment_id,
        "file_path": decoded,
        "content": target.read_text(encoding="utf-8"),
        "size_bytes": target.stat().st_size,
        "language": "python" if target.suffix == ".py" else "text",
        "created_at": target.stat().st_ctime,
        "phase": state["phase"],
    }
    return response_envelope(True, data=data, request_id=request_id)


@router.get("/research/{experiment_id}/plots/{plot_name}")
async def get_plot(experiment_id: str, plot_name: str):
    logger.info("api.files.plot", experiment_id=experiment_id, plot_name=plot_name)
    try:
        state = await get_experiment_or_404(experiment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Experiment not found")
    target = Path(state["project_path"]) / "outputs" / "plots" / plot_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(target), media_type="image/png")
