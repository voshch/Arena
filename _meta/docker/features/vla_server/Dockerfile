ARG TORCH_TAG=2.5.1-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:${TORCH_TAG} AS base

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# install, patch and pre-fetch CLIP on its own layer so later dependency changes do not re-download it
RUN pip install openai-clip \
    && sed -i 's/from pkg_resources import packaging/import packaging, packaging.version/' \
       "$(python -c 'import clip, os; print(os.path.dirname(clip.__file__))')/clip.py" \
    && python -c "import clip; clip.load('ViT-B/32', device='cpu')"

RUN pip install fastapi "uvicorn[standard]" Pillow numpy requests packaging matplotlib efficientnet_pytorch python-multipart pyyaml