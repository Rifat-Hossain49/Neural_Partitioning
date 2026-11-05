FROM registry.codeocean.com/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    matplotlib==3.11.0 \
    numpy==2.4.2 \
    pandas==3.0.1 \
    scipy==1.18.0 \
    torch==2.13.0 \
    tqdm==4.67.3
