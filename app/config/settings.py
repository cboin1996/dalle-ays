import os
import sys
from datetime import datetime
from typing import List, Optional

import pydantic
from pydantic import BaseModel, BaseSettings


class DalleConfig(BaseSettings):
    """Configuration using .env file or defaults declared in here"""

    dalle_model_str: str = "dalle-mini/dalle-mini/mega-1-fp16:latest"
    dalle_commit_id: str = ""
    vqgan_repo: str = "dalle-mini/vqgan_imagenet_f16_16384"
    vqgan_commit_id: str = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
    default_num_pics: int = 3
    wandb_api_key: str
    root_dir: str = sys.path[0]
    model_dir: str = os.path.join(root_dir, "models")
    dalle_bart_model_dir_template: str = os.path.join(model_dir, "%s", "dalle-bart-%s")
    # JAX memory settings https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    xla_python_client_mem_fraction: float = 0.85
    xla_python_client_preallocate: bool = False
    xla_python_client_allocator: Optional[str] = "platform"

    dalle_bart_tokenizer_dir_template: str = os.path.join(
        model_dir, "%s", "dalle-bart-tokenizer-%s"
    )
    dalle_bart_config_dir_template: str = os.path.join(
        model_dir, "%s", "dalle-bart-config-%s"
    )
    vqgan_model_dir_template: str = os.path.join(model_dir, "%s", "vqgan-%s")
    outputs_dir: str = os.path.join(root_dir, "outputs")
    dirs: List[str] = [model_dir, outputs_dir]
    log_level: str
    image_format: str = ".png"

    class Config:
        config_path = os.path.join(os.path.dirname(sys.path[0]), ".env")
        env_file = config_path
        env_file_encoding = "utf-8"

    def get_formatted_dalle_bart_model_dir(self, id: int, revision: str) -> str:
        if revision == "":
            revision = "None"
        return self.dalle_bart_model_dir_template % (id, revision)

    def get_formatted_dalle_bart_tokenizer_dir(self, id: int, revision: str) -> str:
        if revision == "":
            revision = "None"
        return self.dalle_bart_tokenizer_dir_template % (id, revision)

    def get_formatted_dalle_bart_config_dir(self, id: int, revision: str) -> str:
        if revision == "":
            revision = "None"
        return self.dalle_bart_config_dir_template % (id, revision)

    def get_formatted_vqgan_model_dir(self, id: int, revision: str) -> str:
        if revision == "":
            revision = "None"
        return self.vqgan_model_dir_template % (id, revision)
