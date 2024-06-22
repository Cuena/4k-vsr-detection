# Use the nvidia/cuda base image with the specified version
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables (add any required ones here)
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /synthetic-detector
# Run commands to set up the environment
RUN apt update -y && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y git && \
    apt install -y curl && \
    apt install -y python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    apt install -y python3.10-venv python3.10-dev && \
    curl -Ss https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    \
    # Install Python packages
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install lightning==2.2.3 && \
    pip3 install torchmetrics==1.3.2 && \
    pip3 install hydra-core==1.3.2 && \
    pip3 install hydra-colorlog==1.2.0 && \
    pip3 install hydra-optuna-sweeper==1.2.0 && \
    pip3 install protobuf==4.25.2 && \
    pip3 install wandb==0.16.6 && \
    pip3 install mlflow --ignore-installed && \
    pip3 install aim==3.19.3 && \
    pip3 install rootutils && \
    pip3 install pre-commit && \
    pip3 install rich && \
    pip3 install pytest && \
    pip3 install opencv-python==4.9.0.80 && \
    pip3 install albumentations==1.4.4 && \
    pip3 install timm==1.0.3

COPY . /synthetic-detector
CMD ["python", "-v"]
ENTRYPOINT []


