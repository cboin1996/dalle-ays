import logging
import os
from functools import lru_cache

from fastapi import Depends

from .config import settings


# Any common endpoints can be placed here
@lru_cache()
def get_dalle_settings() -> settings.DalleConfig:
    """Get dalle settings class"""
    return settings.DalleConfig()
