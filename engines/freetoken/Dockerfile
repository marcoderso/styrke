# FreeToken (MoE-Engine mit Experten-Auslagerung) für styrke
#
# Bauen:  docker build -t styrke/freetoken:0.2.1 engines/freetoken
#
# FreeToken braucht Treiber r580+ und ein CUDA-13-Toolkit mit nvcc, weil es
# seine Kernels beim ersten Start übersetzt. Der Kernel-Cache landet im
# eingehängten Cache-Ordner (host.cache_dir in der Testmatrix), damit das nur
# einmal je GPU-Typ passiert. Einen Cache von einer anderen GPU-Generation
# nicht übernehmen.

ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

# TORCH_CUDA_ARCH_LIST: 9.0 = Hopper (H100/H200), 12.0 = Blackwell.
ARG CUDA_ARCH=9.0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/ft \
    PATH=/opt/ft/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git curl ca-certificates numactl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Version festgenagelt, damit Messungen wiederholbar bleiben.
ARG FREETOKEN_SPEC="freetoken[accel]==0.2.1"
RUN uv venv /opt/ft --python 3.12 \
    && uv pip install --python /opt/ft/bin/python "${FREETOKEN_SPEC}" "huggingface_hub[cli]" \
    && ft --version

EXPOSE 1919
ENTRYPOINT ["ft"]
CMD ["serve", "--help"]
