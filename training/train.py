import os
os.environ["BNB_CUDA_VERSION"] = "129"

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
TRAIN_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "train.json"))
VAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "val.json"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-LoRA"))

class ChatDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=1024, max_samples=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return min(len(self.data), self.max_samples)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        
        # Apply Qwen2.5 chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Labels are same as input_ids for causal LM (shift is done inside PyTorch loss calculation)
        labels = input_ids.copy()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Configuring QLoRA 4-bit parameters...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config targeting Qwen attention & projection layers
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("Loading datasets...")
    train_dataset = ChatDataset(TRAIN_DATA_PATH, tokenizer, max_length=512, max_samples=100000)
    val_dataset = ChatDataset(VAL_DATA_PATH, tokenizer, max_length=512, max_samples=200)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        max_steps=1200, # Run 1200 steps to optimize
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    
    print("Starting training...")
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoint-1000")
    if os.path.exists(checkpoint_dir):
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    
    print("Saving adapter model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SUCCESS] Model trained and adapters saved at {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
