# DOCKERFILE to RUN distributed training in Google Cloud

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/London \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python3.9 \
         python3-distutils \
         python3-apt \
         python3-dev gcc libc-dev build-essential && \
     rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

RUN pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch/

# Installs pytorch geometric for pytorch-1.12 with CUDA 11.3
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# Install dgl with CUDA 11.3
RUN pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html

# Install pytorch-lightning
RUN pip install pytorch-lightning

RUN pip install scikit-learn tqdm
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
RUN mkdir /trainer

COPY docker_logic_2.py /trainer/main.py
COPY ogbn_products /trainer/ogbn_products

WORKDIR /trainer

RUN pip install ogb 
# Setups the entry point to invoke the trainer.
# RUN pytorch-geometric
ENTRYPOINT ["python3", "main.py"]