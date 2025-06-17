import os
import logging
import torch
import tarfile
import requests
import mimetypes
from llmcompressor import oneshot
from llmcompressor.modifiers import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QUANTIZATION_METHOD = os.getenv("QUANTIZATION_METHOD", "auto-round")
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise ValueError("FATAL: Environment variable MODEL_ID is not set.")

BITS = int(os.getenv("BITS", "4"))
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "128"))
FORMATS = os.getenv("FORMATS", "auto_round")
OUTPUT_DIR = "/app/output"

UPLOAD_AFTER_DONE = os.getenv("UPLOAD_AFTER_DONE", "false").lower() == 'true'
UPLOAD_TYPE = os.getenv("UPLOAD_TYPE", "local")
ZIPLINE_DOMAIN = os.getenv("ZIPLINE_DOMAIN")
ZIPLINE_TOKEN = os.getenv("ZIPLINE_TOKEN")

def compress_directory(output_filename, source_dir):
    logging.info(f"Compressing {source_dir} to {output_filename}...")
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    logging.info("Compression complete.")
    return output_filename

def upload_to_zipline(file_path):
    if not ZIPLINE_DOMAIN or not ZIPLINE_TOKEN:
        logging.error("Zipline domain or token is missing. Cannot upload.")
        return False

    upload_url = f"https://{ZIPLINE_DOMAIN}/api/upload"
    logging.info(f"Uploading {file_path} to Zipline at {upload_url}...")

    try:
        headers = {'authorization': ZIPLINE_TOKEN}
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, mime_type)}
            response = requests.post(upload_url, headers=headers, files=files)
            response.raise_for_status()

        response_data = response.json()
        file_url = response_data.get("files", [{}])[0].get("url")
        if file_url:
            logging.info(f"Successfully uploaded to Zipline! URL: {file_url}")
            return True
        else:
            logging.error(f"Zipline upload failed. Server response was malformed: {response.text}")
            return False

    except Exception as e:
        logging.error(f"An error occurred during Zipline upload: {e}", exc_info=True)
        return False

def main():
    logging.info(f"--- Starting Quantization for: {MODEL_ID} ---")

    try:
        logging.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        logging.info("Model and tokenizer loaded.")

        model_name = MODEL_ID.split('/')[-1]

        if QUANTIZATION_METHOD == "llm-compressor":
            logging.info(f"--- Using llm-compressor for quantization ---")
            # Define the output path based on the new library and scheme
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}-W4A16-llmcompressor")
            logging.info(f"Quantizing model with llm-compressor (GPTQ W4A16)...")

            # Define the llm-compressor recipe for W4A16 GPTQ
            # Targeting all Linear layers, common practice is to ignore lm_head
            recipe = [
                GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),
            ]

            # Apply quantization using llmcompressor.oneshot
            # Using "open_platypus" as a default calibration dataset
            # Using common defaults for num_calibration_samples and max_seq_length
            oneshot(
                model=model,  # Pass the loaded model object
                dataset="open_platypus",
                recipe=recipe,
                output_dir=output_path,
                max_seq_length=2048,
                num_calibration_samples=512,
            )

            logging.info(f"Saving tokenizer to {output_path}...")
            # llmcompressor.oneshot handles model saving. We still need to save the tokenizer.
            # Ensure the output directory exists (oneshot might create it, but good to be sure)
            os.makedirs(output_path, exist_ok=True)
            tokenizer.save_pretrained(output_path)
            logging.info("--- llm-compressor quantization and saving complete. ---")
        elif QUANTIZATION_METHOD == "auto-round":
            logging.info(f"--- Using auto-round for quantization ---")
            autoround = AutoRound(model, tokenizer, bits=BITS, group_size=GROUP_SIZE)
            output_path = os.path.join(OUTPUT_DIR, f"{model_name}-{BITS}bit") # auto-round uses BITS from env
            logging.info(f"Starting quantization and saving to {output_path}...")
            autoround.quantize_and_save(output_path, format=FORMATS)
            logging.info("--- Auto-round quantization and saving complete. ---")
        else:
            raise ValueError(f"Unsupported QUANTIZATION_METHOD: {QUANTIZATION_METHOD}")

        if UPLOAD_AFTER_DONE:
            archive_path = f"{output_path}.tar.gz"
            compress_directory(archive_path, output_path)

            if UPLOAD_TYPE == "zipline":
                upload_successful = upload_to_zipline(archive_path)
                if upload_successful:
                    logging.info(f"Upload successful. Removing local archive: {archive_path}")
                    os.remove(archive_path)
                else:
                    logging.warning(f"Upload failed. Local archive kept at: {archive_path}")
            elif UPLOAD_TYPE == "local":
                logging.info(f"UPLOAD_TYPE is 'local'. Keeping compressed archive at {archive_path}.")
            else:
                logging.warning(f"Unknown UPLOAD_TYPE '{UPLOAD_TYPE}'. No action taken on archive.")
        else:
            logging.info("UPLOAD_AFTER_DONE is false. Skipping compression and upload.")

    except Exception as e:
        logging.error(f"A critical error occurred in main function: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
