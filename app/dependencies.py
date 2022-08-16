import datetime
import glob
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from flax.jax_utils import (
    replicate,
)
import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from dalle_mini.model import DalleBartConfig, DalleBartTokenizer  # Dalle models
from fastapi import Depends, HTTPException, Request, params
from fastapi.logger import logger
from pydantic import BaseModel
from vqgan_jax.modeling_flax_vqgan import VQModel  # vqgan model

from .config import settings

uvicorn_logger = logging.getLogger("uvicorn.error")
logger.handlers = uvicorn_logger.handlers
logger.setLevel(uvicorn_logger.level)


class ParameterizedDepends(params.Depends):
    """Parameterized dependency for use in advanced sub dependencies:
    Ref: https://github.com/tiangolo/fastapi/issues/1655#issuecomment-742293634
    """

    def __init__(self, dependency) -> None:
        self.real_dependency = dependency
        super().__init__(self.__call__)

    async def __call__(self, request: Request):
        async def internal():
            param = yield
            yield await get(
                request.app,
                lambda item=Depends(self.real_dependency(param)): item,
                request,
            )

        value = internal()
        await value.asend(None)
        return value


# Any common endpoints can be placed here
@lru_cache()
def get_dalle_settings() -> settings.DalleConfig:
    """Get dalle settings class"""
    return settings.DalleConfig()


class ModelPaths(BaseModel):
    """Model for path structure that represents how
    dalle and its models are stored
    """

    dalle: str = ""
    vqgan: str = ""
    dalle_processor_tokenizer: str = ""
    dalle_processor_config: str = ""

    def __bool__(self):
        return (
            self.dalle != ""
            and self.vqgan != ""
            and self.dalle_processor_tokenizer != ""
            and self.dalle_processor_config != ""
        )

    def __hash__(self):
        return hash(
            (
                self.dalle,
                self.vqgan,
                self.dalle_processor_tokenizer,
                self.dalle_processor_config,
            )
        )

    def __eq__(self, other):
        return (
            self.dalle == other.dalle
            and self.vqgan == other.vqgan
            and self.dalle_processor_tokenizer == other.dalle_processor_tokenizer
            and self.dalle_processor_config == self.dalle_processor_config
        )


def valid_paths(*args: str) -> bool:
    """Confirm that the given paths exist

    Args:
        *args (str): generic number of path arguments

    Returns:
        bool: true if the given paths exist
    """
    for path in args:
        if not os.path.exists(path):
            return False

    return True

class ImageSearchResponse(BaseModel):
    images: List[str]=[]

def image_browser(
    settings: settings.DalleConfig = Depends(get_dalle_settings),
    search_param: str = None,
    starts_with: bool = False
) -> ImageSearchResponse:
    """Browse and return all the image paths on disk

    Returns:
        List[str] : the images
    """
    if search_param:
        if starts_with:
            paths = [filename for filename in os.listdir(settings.outputs_dir) if filename.startswith(search_param)]  
        paths = glob.glob(os.path.join(settings.outputs_dir, f'*{search_param}*{settings.image_format}'))
            
    else:
        paths = glob.glob(os.path.join(settings.outputs_dir, f'*{settings.image_format}'))
    return ImageSearchResponse(images=paths)

def model_browser(
    settings: settings.DalleConfig = Depends(get_dalle_settings),
    dalle_sha: Optional[str] = None,
    vqgan_sha: Optional[str] = None,
) -> ModelPaths:
    """Browse local files for models matching the dalle_sha and vqgan_sha

    Args:
        dalle_sha (Optional[str], optional): the dalle sha (can be loaded from wandb). Defaults to None.
        vqgan_sha (Optional[str], optional): the vqgan sha (can be loaded from wandb). Defaults to None.

    Returns:
        ModelPaths: the ModelPaths object containing references to local model values if any are found.
    """
    if dalle_sha is None:
        dalle_sha = settings.dalle_commit_id
    if vqgan_sha is None:
        vqgan_sha = settings.vqgan_commit_id

    # create the glob patterns
    for model_id_path in glob.glob(os.path.join(settings.model_dir, "*")):
        # parse out the id from the path
        _id = os.path.split(model_id_path)[-1]
        dalle_path = settings.get_formatted_dalle_bart_model_dir(_id, dalle_sha)
        vqgan_path = settings.get_formatted_vqgan_model_dir(_id, vqgan_sha)
        dalle_tokenizer_path = settings.get_formatted_dalle_bart_tokenizer_dir(
            _id, dalle_sha
        )
        dalle_config_path = settings.get_formatted_dalle_bart_config_dir(_id, dalle_sha)

        if valid_paths(dalle_path, vqgan_path, dalle_tokenizer_path, dalle_config_path):
            return ModelPaths(
                dalle=dalle_path,
                vqgan=vqgan_path,
                dalle_processor_tokenizer=dalle_tokenizer_path,
                dalle_processor_config=dalle_config_path,
            )

    return ModelPaths()


class DalleModelObject:
    """Object for storing the loaded dalle model, vqgan and processor"""

    dalle_mini: DalleBart
    dalle_mini_params: object
    repl_dalle_mini_params: object
    processor: DalleBartProcessor
    vqgan: VQModel
    vqgan_params: object
    repl_vqgan_params: object


@lru_cache(maxsize=1)
def model_loader(model_paths: ModelPaths) -> DalleModelObject:
    """Load model

    Args:
        model_paths (ModelPaths): the paths object for loading the model.

    Returns:
        DalleModelObject: the object containing the dalle model, processor and vqgan model
    """
    # load local model if exists
    if not valid_paths(
        model_paths.dalle,
        model_paths.dalle_processor_config,
        model_paths.dalle_processor_tokenizer,
        model_paths.vqgan,
    ):
        raise HTTPException(
            status_code=404, detail=f"No valid model found for paths {model_paths}"
        )
    logger.info(f"Attempting to load models at {model_paths}")
    dalle_obj = DalleModelObject()

    # Load the processor from disk, by building up the tokenizer, config and THEN the processor :9
    logger.debug(
        f"Loading dalle-mini tokenizer from {model_paths.dalle_processor_tokenizer}"
    )
    tokenizer = DalleBartTokenizer.from_pretrained(
        model_paths.dalle_processor_tokenizer
    )
    logger.debug(f"Loading dalle-mini config from {model_paths.dalle_processor_config}")
    config = DalleBartConfig.from_pretrained(model_paths.dalle_processor_config)

    logger.info(f"Constructing processor from config and tokenizer!")
    processor = DalleBartProcessor(
        tokenizer, config.normalize_text, config.max_text_length
    )
    dalle_obj.processor = processor

    # Load dalle mini
    logger.info(f"loading dalle-mini from {model_paths.dalle}!")
    dalle_obj.dalle_mini, dalle_obj.dalle_mini_params = DalleBart.from_pretrained(
        model_paths.dalle,
        dtype=jnp.float32,
        _do_init=False,
    )

    # Load VQGAN
    logger.info(f"Loading vqgan from {model_paths.vqgan}")
    dalle_obj.vqgan, dalle_obj.vqgan_params = VQModel.from_pretrained(
        model_paths.vqgan, _do_init=False
    )

    # Model parameters are replicated on each device for faster inference.
    dalle_obj.repl_dalle_mini_params = replicate(dalle_obj.dalle_mini_params)
    dalle_obj.repl_vqgan_params = replicate(dalle_obj.vqgan_params)

    return dalle_obj


def get_formatted_image_name(prompt: str, idx: int, uid: str) -> str:
    return "%s %d %s.png" % (prompt, idx, uid)
