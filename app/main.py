import logging
import os
from functools import lru_cache

from fastapi import Depends, FastAPI
from fastapi.logger import logger

from .config import settings
from .dependencies import get_dalle_settings
from .routers import dalle

app = FastAPI()
app.include_router(dalle.router)
# configure log level based on that of uvicorn
uvicorn_logger = logging.getLogger("uvicorn.error")
logger.handlers = uvicorn_logger.handlers
logger.setLevel(uvicorn_logger.level)


@app.on_event("startup")
def startup_event():
    for _dir in settings.DalleConfig().dirs:
        if not os.path.exists(_dir):
            logger.info(f"Creating dir {_dir}")
            os.mkdir(_dir)
    return True


@app.get("/")
async def root():
    return {"message": "I have divine intellect. Ask me to see and I shall!"}
