import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Get Config from Environment ---
MODEL_ID = os.getenv("MODEL_ID")
if not MODEL_ID:
    raise ValueError("FATAL: Environment variable MODEL_ID is not set.")

BITS = int(os.getenv("BITS", "4"))
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "128"))
FORMATS = os.getenv("FORMATS", "auto_round")
RECIPE = os.getenv("RECIPE", "auto-round")
LOW_MEM = os.getenv("LOW_MEM", "false").lower() == "true"
OUTPUT_DIR = "/app/output"

def main():
    logging.info(f"--- Advanced Loading Attempt for: {MODEL_ID} ---")
    logging.info(f"Using configuration: BITS={BITS}, GROUP_SIZE={GROUP_SIZE}, FORMATS={FORMATS}")

    try:
        # --- Loading Model ---
        # device_map="auto" will now see both GPUs and distribute the model across them
        logging.info("Loading model with trust_remote_code=True across all available GPUs...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        logging.info("Model and tokenizer loaded successfully.")

        # --- Quantization ---
        logging.info(f"Initializing AutoRound with '{RECIPE}' recipe...")

        # We let accelerate handle device placement.
        # The `device` parameter is removed from the constructor.
        autoround = AutoRound(
            model,
            tokenizer,
            bits=BITS,
            group_size=GROUP_SIZE,
            sym=True,
            recipe=RECIPE,
            low_gpu_mem_usage=LOW_MEM
        )

        output_path = os.path.join(OUTPUT_DIR, f"{MODEL_ID.split('/')[-1]}-{BITS}bit")

        logging.info(f"Starting quantization and saving to {output_path}...")
        autoround.quantize_and_save(output_path, format=FORMATS)

        logging.info("--- All done! Model quantized successfully. ---")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
