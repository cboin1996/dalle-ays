import datetime
import glob
import logging
import os
import random
from functools import (  # Model functions are compiled and parallelized to take advantage of multiple devices.
    lru_cache, partial)
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor  # Load models & tokenizer
from dalle_mini.model import (DalleBartConfig,  # Dalle models
                              DalleBartTokenizer)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.logger import logger
from fastapi.responses import FileResponse
from flax.jax_utils import \
    replicate  # Model parameters are replicated on each device for faster inference.
from flax.training.common_utils import shard_prng_key
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel

from ..config import settings
from ..dependencies import (DalleModelObject, ModelPaths, get_dalle_settings,
                            get_formatted_image_name, image_browser,
                            model_browser, model_loader)

uvicorn_logger = logging.getLogger("uvicorn.error")
logger.handlers = uvicorn_logger.handlers
logger.setLevel(uvicorn_logger.level)

router = APIRouter(
    prefix="/dalle",
    tags=["dalle"],
)


@router.get("/browse", response_model=ModelPaths)
def browse(
    vqgan_sha: Optional[str] = None,
    dalle_sha: Optional[str] = None,
    model_paths: ModelPaths = Depends(model_browser),
):
    """Browses saved models for matching dalle and vqgan sha, returning the id."""
    return model_paths


@router.get("/images", response_model=List[str])
def images(image_paths: List[str] = Depends(image_browser)):
    """Get a list of all images on disk.

    Returns:
        List[str]: the list of images
    """
    if len(image_paths) == 0:
        return HTTPException(
            404, detail=f"No images found! Try /dalle/show to generate some images :)"
        )

    return image_paths


@router.get("/image")
def get_image(image_path: str):
    return FileResponse(image_path)


@router.get("/pull", response_model=ModelPaths)
def pull(
    vqgan_sha: Optional[str] = None,
    dalle_sha: Optional[str] = None,
    force: bool = False,
    model_paths: ModelPaths = Depends(model_browser),
    settings: settings.DalleConfig = Depends(get_dalle_settings),
):
    """Pull model from wandb and save to disk, returning the paths to the model"""
    # return local paths if found and not overridden by user
    if model_paths and not force:
        logger.info(f"Model already exists: {model_paths}")
        return model_paths

    if dalle_sha is None:
        dalle_sha = settings.dalle_commit_id
    if vqgan_sha is None:
        vqgan_sha = settings.vqgan_commit_id

    logger.info(f"Pulling dalle-mini: {settings.dalle_model_str}, rev={dalle_sha}")
    # Load dalle-mini
    dalle_mini, dalle_mini_params = DalleBart.from_pretrained(
        settings.dalle_model_str,
        revision=dalle_sha,
        dtype=jnp.float32,
        _do_init=False,
    )

    logger.info(f"Pulling vqgan: {settings.vqgan_repo}, rev={vqgan_sha}")
    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        settings.vqgan_repo, revision=vqgan_sha, _do_init=False
    )

    logger.info(f"Pulling dalle processor: {settings.dalle_model_str}, rev={dalle_sha}")
    # Load processor
    processor = DalleBartProcessor.from_pretrained(
        settings.dalle_model_str,
        revision=dalle_sha,
    )

    logger.info(f"Pulling dalle config: {settings.dalle_model_str}, rev={dalle_sha}")
    # Have to load the config separate, due to no sdk exposure in the processor
    config = DalleBartConfig.from_pretrained(
        settings.dalle_model_str, revision=dalle_sha
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    # construct path object
    paths = ModelPaths(
        dalle=settings.get_formatted_dalle_bart_model_dir(timestamp, dalle_sha),
        vqgan=settings.get_formatted_vqgan_model_dir(timestamp, vqgan_sha),
        dalle_processor_tokenizer=settings.get_formatted_dalle_bart_tokenizer_dir(
            timestamp, dalle_sha
        ),
        dalle_processor_config=settings.get_formatted_dalle_bart_config_dir(
            timestamp, dalle_sha
        ),
    )

    logger.info(f"Saving dalle-mini to: {paths.dalle}")
    # save dalle-mini
    dalle_mini.save_pretrained(
        paths.dalle,
        params=dalle_mini_params,
    )

    # Note: the processor model does not have save ability.. exposed yet,
    # so I do some absolute black magic to save the subcomponents (tokenizer and config)
    # and load them back later :9
    logger.info(
        f"Saving dalle-processor-tokenizer to: {paths.dalle_processor_tokenizer}"
    )
    processor.tokenizer.save_pretrained(paths.dalle_processor_tokenizer)

    logger.info(f"Saving dalle-processor-config to: {paths.dalle_processor_config}")
    config.save_pretrained(paths.dalle_processor_config)

    # save vqgan
    logger.info(f"Saving vqgan to: {paths.vqgan}")
    vqgan.save_pretrained(
        paths.vqgan,
        params=vqgan_params,
    )

    return paths


class ImagePathResponse(BaseModel):
    prompts: Dict[str, List[str]] = {}


@router.post("/show", response_model=ImagePathResponse)
def show(
    queries: List[str],
    n_predictions: int,
    settings: settings.DalleConfig = Depends(get_dalle_settings),
    model_obj: DalleModelObject = Depends(model_loader),
):
    """Query the current model

    Args:
        query (str): the query to process in dalle, returning the generated image as a response
    """
    model = model_obj.dalle_mini
    model_params = model_obj.dalle_mini_params
    processor = model_obj.processor
    vqgan = model_obj.vqgan
    vqgan_params = model_obj.vqgan_params

    logger.info(f"Recieved query: {queries}")
    # check how many devices are available
    logger.debug(
        f"Num. available devices for parrallel processing: {jax.local_device_count()}"
    )
    # Model parameters are replicated on each device for faster inference.
    params = replicate(model_params)
    vqgan_params = replicate(vqgan_params)

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)
    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0
    logger.info(f"Tokenizing queries {queries}")
    # Note: we could use the same prompt multiple times for faster inference.
    tokenized_queries = processor(queries)
    # Finally we replicate the prompts onto each device.
    tokenized_query = replicate(tokenized_queries)

    # generate images
    images = []
    response = ImagePathResponse()
    logger.info(f"Performing inference!")
    for i in tqdm(range(max(n_predictions // jax.device_count(), 1))):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_query,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for query_idx, decoded_img in enumerate(decoded_images):
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            # generate filename
            generated_name = get_formatted_image_name(
                queries[query_idx],
                i,
                datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            )

            logger.debug(f"Generated img with name: {generated_name}")
            path = os.path.join(settings.outputs_dir, generated_name)
            img.save(
                path,
                quality="keep",
            )
            images.append(img)

            # transform prompts into response format
            if queries[query_idx] not in response.prompts:
                response.prompts[queries[query_idx]] = [path]
            else:
                response.prompts[queries[query_idx]].append(path)

    return response
