FROM nvcr.io/nvidia/jax:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    x11-apps \
    x11-xserver-utils \
    python3-tk \
    git \
    curl \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /work/
RUN pip install --no-cache-dir -r requirements.txt
