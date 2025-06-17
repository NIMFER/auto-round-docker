# AutoRound & Quanto Model Quantization App

This application allows you to quantize Hugging Face transformer models using either the AutoRound library or the Quanto library. Quantization helps reduce model size and can speed up inference, with a potential trade-off in accuracy.

## Features

-   Supports two quantization libraries:
    -   **AutoRound**: Offers various recipes for quantization.
    -   **Quanto**: Provides straightforward quantization, currently configured for int4 weight quantization.
-   Dockerized for easy setup and execution.
-   Configurable via environment variables in `docker-compose.yaml`.
-   Option to automatically compress and upload quantized models to Zipline or save them locally.

## Prerequisites

-   Docker
-   Docker Compose
-   NVIDIA GPU with drivers that support CUDA (if using GPU acceleration)

## How to Use

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Configure `docker-compose.yaml`:**
    Open `docker-compose.yaml` and adjust the environment variables under the `autoround-quantizer` service as needed:

    *   `MODEL_ID`: (Required) The Hugging Face model ID (e.g., "Qwen/Qwen2-0.5B") or a path to a local model folder.
    *   `QUANTIZATION_METHOD`: (Optional) Choose the quantization library.
        *   `"auto-round"` (Default): Uses the AutoRound library.
        *   `"quanto"`: Uses the Quanto library (quantizes weights to int4, activations are not quantized).
    *   `RECIPE`: (For `auto-round` only) The AutoRound recipe to use (e.g., "auto-round", "auto-round-best", "auto-round-light"). Default: "auto-round".
    *   `BITS`: (For `auto-round` only) Target bits for quantization with AutoRound (e.g., "4", "8"). Default: "4".
    *   `GROUP_SIZE`: (For `auto-round` only) Quantization group size for AutoRound. Default: "128".
    *   `FORMATS`: (For `auto-round` only) Output formats for AutoRound (comma-separated list, e.g., "auto_awq,auto_gptq,auto_round"). Default: "auto_awq".
    *   `LOW_MEM`: (Optional) Set to "true" to enable low VRAM mode, especially for large models. Default: "true".
    *   `HF_HUB_ETAG_TIMEOUT`: (Optional) Increases download timeout for large files from Hugging Face Hub. Default: "120".

    *   **Upload Control (Optional):**
        *   `UPLOAD_AFTER_DONE`: Set to "true" to enable compression and/or uploading after quantization. Default: "false".
        *   `UPLOAD_TYPE`: Determines the post-quantization action if `UPLOAD_AFTER_DONE` is true.
            *   `"local"` (Default): Compresses the output and keeps the local archive.
            *   `"zipline"`: Compresses, uploads to Zipline, then deletes the local archive.
        *   `ZIPLINE_DOMAIN`: (Required if `UPLOAD_TYPE` is "zipline") Your Zipline instance domain (e.g., "files.example.com").
        *   `ZIPLINE_TOKEN`: (Required if `UPLOAD_TYPE` is "zipline") Your Zipline authorization token.

3.  **Build and Run the Docker Container:**
    Open your terminal in the project's root directory and run:
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image (if it's the first time or if `Dockerfile` changed) and then start the quantization process.

    To run in detached mode (in the background):
    ```bash
    docker-compose up --build -d
    ```

4.  **Accessing Quantized Models:**
    -   The quantized models will be saved in the `./autoround_output` directory on your host machine (this is mapped to `/app/output` inside the container).
    -   The output directory structure will be:
        -   For AutoRound: `./autoround_output/<model_name>-<BITS>bit/`
        -   For Quanto: `./autoround_output/<model_name>-int4-quanto/`
    -   If `UPLOAD_AFTER_DONE` is true and `UPLOAD_TYPE` is "local", a compressed `.tar.gz` archive will also be found in `./autoround_output/`.

## Example Usage

Assuming you want to quantize `Qwen/Qwen2-0.5B` using the `quanto` method:

1.  Set `MODEL_ID: "Qwen/Qwen2-0.5B"` in `docker-compose.yaml`.
2.  Set `QUANTIZATION_METHOD: "quanto"` in `docker-compose.yaml`.
3.  Run `docker-compose up --build`.
4.  Find the quantized model in `./autoround_output/Qwen2-0.5B-int4-quanto/`.

## GPU Allocation

The `docker-compose.yaml` is configured to allocate GPU resources using the Container Device Interface (CDI).
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: cdi
          device_ids: ['nvidia.com/gpu=all'] # Use 'all' or specify GPU IDs e.g., ['0', '1']
          capabilities: ['gpu']
```
Ensure your Docker version and NVIDIA drivers support CDI for GPU access within the container.

## Stopping the Application

-   If running in the foreground (without `-d`), press `Ctrl+C` in the terminal.
-   If running in detached mode, use:
    ```bash
    docker-compose down
    ```

This will stop and remove the container. The output files in `./autoround_output` will persist.
