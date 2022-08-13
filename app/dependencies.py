import datetime
import glob
import logging
import os
from functools import lru_cache
from typing import Dict, Optional

import jax.numpy as jnp
from dalle_mini import DalleBart, DalleBartProcessor
from dalle_mini.model import DalleBartTokenizer, DalleBartConfig # Dalle models
from fastapi import Depends, Request, params, HTTPException
from pydantic import BaseModel
from vqgan_jax.modeling_flax_vqgan import VQModel  # vqgan model

from .config import settings


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
        return self.dalle != "" and self.vqgan != "" and self.dalle_processor_tokenizer != "" and self.dalle_processor_config != ""

    def __hash__(self):
        return hash((self.dalle, self.vqgan, self.dalle_processor_tokenizer, self.dalle_processor_config))

    def __eq__(self, other):
        return (
            self.dalle == other.dalle
            and self.vqgan == other.vqgan
            and self.dalle_processor_tokenizer == other.dalle_processor_tokenizer
            and self.dalle_processor_config == self.dalle_processor_config
        )


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
        dalle_tokenizer_path = settings.get_formatted_dalle_bart_tokenizer_dir(_id, dalle_sha)
        dalle_config_path = settings.get_formatted_dalle_bart_config_dir(_id, dalle_sha)

        if os.path.exists(dalle_path) and os.path.exists(vqgan_path) \
          and os.path.exists(dalle_tokenizer_path) and os.path.exists(dalle_config_path):
            return ModelPaths(
                dalle=dalle_path,
                vqgan=vqgan_path,
                dalle_processor_tokenizer=dalle_tokenizer_path,
                dalle_processor_config=dalle_config_path
            )

    return ModelPaths()


class DalleModelObject:
    """Object for storing the loaded dalle model, vqgan and processor"""

    dalle_mini: DalleBart
    dalle_mini_params: object
    processor: DalleBartProcessor
    vqgan: VQModel
    vqgan_params: object


@lru_cache()
def model_loader(model_paths: ModelPaths) -> DalleModelObject:
    """Load model

    Args:
        model_paths (ModelPaths): the paths object for loading the model.

    Returns:
        DalleModelObject: the object containing the dalle model, processor and vqgan model
    """
    # load local model if exists
    if not model_paths:
        raise HTTPException(
            status_code=404, detail=f"No valid model found for paths {model_paths}"
        )

    dalle_obj = DalleModelObject()

    # Load the processor from wandb
    # processor_revision = model_paths.dalle_processor.split(":")[1]
    # dalle_obj.processor = DalleBartProcessor.from_pretrained(
    #     model_paths.dalle_processor,
    #     revision=processor_revision,
    # )

    # Load the processor from disk, by building up the tokenizer, config and THEN the processor :9
    tokenizer = DalleBartTokenizer.from_pretrained(model_paths.dalle_processor_tokenizer)
    config    = DalleBartConfig.from_pretrained(model_paths.dalle_processor_config)
    processor = DalleBartProcessor(tokenizer, config.normalize_text, config.max_text_length)
    dalle_obj.processor = processor

    # Load dalle mini
    dalle_obj.dalle_mini, dalle_obj.dalle_mini_params = DalleBart.from_pretrained(
        model_paths.dalle,
        dtype=jnp.float16,
        _do_init=False,
    )

    # Load VQGAN
    dalle_obj.vqgan, dalle_obj.vqgan_params = VQModel.from_pretrained(
        model_paths.vqgan, _do_init=False
    )

    return dalle_obj
