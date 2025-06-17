FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && \
    pip install --no-cache-dir \
    "transformers==4.52.4" \
    "tokenizers==0.21.1" \
    "datasets==3.6.0" \
    "trl==0.18.1" \
    "auto-round[all]" \
    "accelerate" \
    "llmcompressor" \
    "vllm" && \
    rm -rf /var/lib/apt/lists/*

COPY run_advanced.py .

RUN mkdir -p /app/output

CMD ["python", "run_advanced.py"]
