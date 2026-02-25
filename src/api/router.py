from fastapi import APIRouter

from src.api.routes.chat import router as chat_router
from src.api.routes.files import router as files_router
from src.api.routes.research import router as research_router
from src.api.routes.system import router as system_router

api_router = APIRouter()
api_router.include_router(research_router, tags=["research"])
api_router.include_router(files_router, tags=["files"])
api_router.include_router(system_router, tags=["system"])
api_router.include_router(chat_router, tags=["chat"])
