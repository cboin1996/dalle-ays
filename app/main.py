import logging
import os
from functools import lru_cache
from logging.config import dictConfig

from fastapi import Depends, FastAPI

from .config import settings
from .dependencies import get_dalle_settings
from .routers import dalle

dictConfig(settings.LogConfig().dict())
logger = logging.getLogger(__name__)
app = FastAPI()
app.include_router(dalle.router)

@app.on_event("startup")
def startup_event():
    for _dir in settings.DalleConfig().dirs:
        if not os.path.exists(_dir):
            os.mkdir(_dir)
    return True


@app.get("/")
async def root():
    logger.info("hi")
    return {"message": "I have divine intellect. Ask me to see and I shall!"}
