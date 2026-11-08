"""
Zenthi-AI OS: Milestone 2 Multi-Adapter QLoRA Training Pipeline
Implements fine-tuning for three discrete, decoupled LoRA task vectors:
1. adapter-code: Code, algorithms, SQL, and type systems.
2. adapter-extraction: HTML-to-JSON, web scraping, and document generation.
3. adapter-vision: OCR transcription, layout parsing, and diagram metrics.

Features:
- Sub-4GB VRAM ceiling with 4-bit NF4 double-quantization + gradient checkpointing.
- Assistant token loss masking (prompts masked with -100) for high-accuracy instruction adherence.
- Per-step checkpointing (every 100 steps) to prevent mid-epoch crash losses.
- Domain filtering on datasets/zenthi_hybrid_train.jsonl.
- Native compatibility with local Qwen 2.5/3.5 architectures.
"""

import os
import sys

# Prevent CUDA memory fragmentation and enable dynamic segment reuse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import gc
import json
import time
import argparse
import torch

# Hardware Thermal & Power Protection Gate:
# 1. Cap PyTorch CPU threads to 4 threads (leaves 20 threads completely free for Brave browser & Windows OS)
torch.set_num_threads(4)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass

# 2. Lower Windows process priority to BELOW_NORMAL so Brave browser gets top priority for video/audio decoding
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

# 3. Reserve 40%+ of GPU VRAM for Brave browser and Windows DWM
# PyTorch is strictly capped to max 60% of total VRAM (~4.89GB), leaving 3.2GB permanently available for Brave
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.60, 0)

from typing import List, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Base Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_DATA_FILE = os.path.join(DATASETS_DIR, "zenthi_hybrid_train.jsonl")
EVAL_DATA_FILE = os.path.join(DATASETS_DIR, "zenthi_hybrid_eval.jsonl")

# Base model on HuggingFace Hub / local cache (Qwen 3B coder backbone)
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Target domain mappings for decoupled adapters
ADAPTER_DOMAINS = {
    "code": ["code", "reasoning"],
    "extraction": ["web_extraction", "doc_generation"],
    "vision": ["ocr_vision", "diagram_analysis"]
}


class DomainChatDataset(Dataset):
    """Loads and formats ChatML conversation pairs filtered by adapter domain,
    masking prompt/system tokens with -100 so loss is computed strictly on assistant responses.
    """
    def __init__(self, jsonl_path: str, target_domains: List[str], tokenizer, max_length: int = 1024, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        print(f"[Dataset] Loading from {os.path.basename(jsonl_path)} for domains: {target_domains}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    domain = item.get("domain", "")
                    if domain in target_domains:
                        self.samples.append(item)
                        if max_samples and len(self.samples) >= max_samples:
                            break
                except json.JSONDecodeError:
                    continue

        print(f"[Dataset] Loaded {len(self.samples)} samples for domains: {target_domains}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]

        # Format with tokenizer's chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Mask prompt tokens with -100 so loss is computed only on assistant turns
        labels = [-100] * len(input_ids)
        # ChatML assistant header: <|im_start|>assistant\n -> [151644, 77091, 198]
        # End token: <|im_end|> -> 151645
        assistant_header = [151644, 77091, 198]
        im_end = 151645

        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i:i + 3] == assistant_header:
                start_resp = i + 3
                end_resp = start_resp
                while end_resp < n and input_ids[end_resp] != im_end:
                    end_resp += 1
                if end_resp < n and input_ids[end_resp] == im_end:
                    end_resp += 1  # include <|im_end|> token in target
                for j in range(start_resp, min(end_resp, n)):
                    labels[j] = input_ids[j]
                i = end_resp
            else:
                i += 1

        # Fallback if no assistant header was matched
        if all(x == -100 for x in labels):
            labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def build_bnb_config():
    """Configures 4-bit NF4 double-quantization for sub-4GB VRAM footprint."""
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )


class ThermalAndPowerThrottleCallback(TrainerCallback):
    """
    Prevents laptop emergency power cuts and thermal runaway:
    1. Enforces brief thermal pacing cool-down pause (0.15s) between steps so heat pipes shed heat.
    2. Periodically calls gc.collect() and empty_cache() to prevent VRAM fragmentation.
    """
    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(0.15)
        if state.global_step % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_adapter(adapter_name: str, base_model_id: str, epochs: int, max_samples: int, batch_size: int, learning_rate: float):
    if adapter_name not in ADAPTER_DOMAINS:
        raise ValueError(f"Unknown adapter name: {adapter_name}. Choose from: {list(ADAPTER_DOMAINS.keys())}")

    target_domains = ADAPTER_DOMAINS[adapter_name]
    output_dir = os.path.join(MODELS_DIR, f"adapter-{adapter_name}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  Training Task Vector: adapter-{adapter_name}")
    print(f"  Base Model: {base_model_id}")
    print(f"  Target Domains: {target_domains}")
    print(f"  Output Directory: {output_dir}")
    print(f"=======================================================")

    # Free up any lingering CUDA memory before allocation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load Tokenizer
    print(f"\n[1/5] Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Datasets (max_length=512 saves 75% attention VRAM, reserving memory for Brave)
    print(f"\n[2/5] Preparing filtered datasets...")
    train_dataset = DomainChatDataset(
        jsonl_path=TRAIN_DATA_FILE,
        target_domains=target_domains,
        tokenizer=tokenizer,
        max_length=512,
        max_samples=max_samples
    )

    # 3. Load 4-Bit Base Model with QLoRA
    print(f"\n[3/5] Loading base model in 4-bit NF4...")
    bnb_config = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Required when gradient checkpointing is active

    # 4. Configure LoRA
    print(f"\n[4/5] Attaching LoRA adapter...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments with Per-Step Checkpointing & Thermal Protection
    print(f"\n[5/5] Configuring trainer with per-step checkpointing...")
    is_bf16 = torch.cuda.is_bf16_supported()
    grad_accum = 8 if batch_size == 1 else max(1, 8 // batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=100,             # Checkpoint every 100 steps (anti-crash safeguard)
        save_total_limit=2,         # Retain only the last 2 checkpoints to conserve disk space
        save_only_model=True,       # Conserve disk footprint
        eval_strategy="no",         # Evaluation is benchmarked via scripts/evaluate_adapters.py to preserve VRAM
        fp16=not is_bf16,
        bf16=is_bf16,
        optim="paged_adamw_8bit",    # 8-bit optimizer reduces VRAM by another ~500MB
        dataloader_pin_memory=False, # Stable on Windows multi-threaded execution
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[ThermalAndPowerThrottleCallback()]
    )

    # Auto-resume from latest checkpoint if available
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_checkpoint = True
            print(f"[Resume] Found checkpoint in {output_dir}. Resuming from latest checkpoint...")

    print("\nStarting QLoRA training run...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Free optimizer and intermediate backward buffers before saving weights to prevent CPU RAM spike
    del trainer.optimizer
    del trainer.lr_scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Saving trained adapter weights to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "adapter_name": f"adapter-{adapter_name}",
        "base_model": base_model_id,
        "target_domains": target_domains,
        "samples_trained": len(train_dataset),
        "epochs": epochs,
        "lora_r": 16,
        "lora_alpha": 32
    }
    with open(os.path.join(output_dir, "adapter_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SUCCESS] Milestone 2: adapter-{adapter_name} training completed successfully.")

    # Cleanup memory
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Zenthi-AI Multi-Adapter Training Pipeline")
    parser.add_argument("--adapter", type=str, default="code", choices=["code", "extraction", "vision", "all"],
                        help="Which decoupled adapter to train")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to train per adapter")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size (default 1 for sub-3.5GB VRAM & thermal safety)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    if args.adapter == "all":
        for a in ["code", "extraction", "vision"]:
            train_adapter(a, args.base_model, args.epochs, args.max_samples, args.batch_size, args.lr)
    else:
        train_adapter(args.adapter, args.base_model, args.epochs, args.max_samples, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
