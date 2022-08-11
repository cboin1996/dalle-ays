FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  git \
  python3.10 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip \
  && pip install fastapi==0.79.0 uvicorn==0.18.2 pydantic==1.9.0 \
  && pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  && pip install -q git+https://github.com/borisdayma/dalle-mini@main \
  && pip install -q git+https://github.com/patil-suraj/vqgan-jax@main \
  && pip install pydantic[dotenv]

WORKDIR /app

RUN useradd --create-home appuser
USER appuser
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
