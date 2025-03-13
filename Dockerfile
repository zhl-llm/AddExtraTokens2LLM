FROM python:3.11-slim

# Set the environment variables
ENV GPTQMODEL_USE_MODELSCOPE="True"
ENV HF_ENDPOINT="https://hf-mirror.com"
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

WORKDIR /app

# Install system dependencies (optional, for some torch features)
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install the dependencies libs
COPY requirements.txt /app/requirements.txt
COPY llm_loader.py /app/llm_loader.py
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "llm_loader.py"]
