import datetime
import glob
import logging
import os
from functools import (  # Model functions are compiled and parallelized to take advantage of multiple devices.
    lru_cache, partial)
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini import DalleBart, DalleBartProcessor  # Load models & tokenizer
from fastapi import APIRouter, Depends, HTTPException
from flax.jax_utils import \
    replicate  # Model parameters are replicated on each device for faster inference.
from flax.training.common_utils import shard_prng_key
from PIL import Image
from pydantic import BaseModel
from tqdm.notebook import trange
from transformers import CLIPProcessor, FlaxCLIPModel
from vqgan_jax.modeling_flax_vqgan import VQModel

from ..config import settings
from ..dependencies import get_dalle_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/dalle",
    tags=["dalle"],
)


class ModelPaths(BaseModel):
    dalle: str = ""
    vqgan: str = ""

    def __bool__(self):
        return self.dalle != "" and self.vqgan != ""


@router.get("/browse")
async def browse(
    vqgan_sha: Optional[str] = None,
    dalle_sha: Optional[str] = None,
    settings: settings.DalleConfig = Depends(get_dalle_settings),
    response_model=ModelPaths,
) -> dict:
    """Browses saved models for matching dalle and vqgan sha, returning the id."""
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
        if os.path.exists(
            settings.get_formatted_dalle_bart_model_dir(_id, dalle_sha)
        ) and os.path.exists(settings.get_formatted_vqgan_model_dir(_id, vqgan_sha)):
            return ModelPaths(dalle=dalle_path, vqgan=vqgan_path)

    return ModelPaths()

def pull(
    vqgan_sha: Optional[str] = None,
    dalle_sha: Optional[str] = None,
    settings: settings.DalleConfig = Depends(get_dalle_settings),
    model_paths: dict = Depends(browse),
):
    if model_paths:
        return {
            f"message": "not downloading new since local models already exist matching vqgan_sha={vqgan_sha}, dalle_sha={dalle_sha}!"
        }

    """Get the stable dalle and vqgan model and store it to disk.. if it already hasnt been grabbed"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if dalle_sha is None:
        dalle_sha = settings.dalle_commit_id
    if vqgan_sha is None:
        vqgan_sha = settings.vqgan_commit_id

    # Load dalle-mini
    model, params = DalleBart.from_pretrained(
        settings.dalle_model,
        revision=settings.dalle_commit_id,
        dtype=jnp.float16,
        _do_init=False,
        api_key=settings.wandb_api_key,
    )

    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        settings.vqgan_repo, revision=settings.vqgan_commit_id, _do_init=False
    )

    # Load the processor model (does not have save ability.. must be loaded at startup)
    processor = DalleBartProcessor.from_pretrained(
        settings.dalle_model,
        revision=settings.dalle_commit_id,
        api_key=settings.wandb_api_key,
    )

    # save dalle-mini
    model.save_pretrained(
        settings.get_formatted_dalle_bart_model_dir(
            timestamp, settings.dalle_commit_id
        ),
        params=params,
    )

    # save vqgan
    vqgan.save_pretrained(
        settings.get_formatted_vqgan_model_dir(timestamp, settings.vqgan_commit_id),
        params=vqgan_params,
    )


@router.get("/show")
def show(query: str, settings: settings.DalleConfig = Depends(get_dalle_settings)):
    """Query the current model

    Args:
        query (str): the query to process in dalle, returning the generated image as a response
    """
    # TODO: Load in the model from app context/memory/? Then see if you can get predictions!
    logger.info("recieved query: {%s}")
    # check how many devices are available
    logger.info(f"Num. Devices: {jax.local_device_count()}")
    # Model parameters are replicated on each device for faster inference.
    params = replicate(params)
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
    # Let's define some text prompts.
    prompts = [query]

    # Note: we could use the same prompt multiple times for faster inference.
    tokenized_prompts = processor(prompts)
    # Finally we replicate the prompts onto each device.
    tokenized_prompt = replicate(tokenized_prompts)

    # generate images
    images = []
    response: Dict[str, List[str]]
    for i in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
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
        for prompt_idx, decoded_img in enumerate(decoded_images):
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            img.save(
                os.path.join(outputs_path, prompts[prompt_idx] + f" {i} .png"),
                quality="keep",
            )
            images.append(img)
            response[prompts[prompt_idx]].append(img)
            print()

    return response
