import os
os.environ["BNB_CUDA_VERSION"] = "129"

import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "HuggingFaceTB/SmolVLM-Instruct"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-Vision-LoRA"))
IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "vision_images"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "vision_train.json"))

def generate_synthetic_vision_dataset():
    """Generates 50 synthetic VQA images and a JSON mapping for offline VLM training."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    dataset = []
    shapes = ["circle", "square", "triangle"]
    colors = ["red", "green", "blue"]
    
    print("Generating synthetic vision dataset...")
    for i in range(50):
        # Pick shape and color
        shape = shapes[i % len(shapes)]
        color = colors[i % len(colors)]
        
        # Create image
        img = Image.new("RGB", (128, 128), color="white")
        draw = ImageDraw.Draw(img)
        
        if shape == "circle":
            draw.ellipse([20, 20, 108, 108], fill=color, outline="black")
        elif shape == "square":
            draw.rectangle([20, 20, 108, 108], fill=color, outline="black")
        elif shape == "triangle":
            draw.polygon([(64, 20), (20, 108), (108, 108)], fill=color, outline="black")
            
        # Save image
        img_name = f"shape_{i}.png"
        img_path = os.path.join(IMAGE_DIR, img_name)
        img.save(img_path)
        
        # VQA details
        dataset.append({
            "image": img_path,
            "conversations": [
                {
                    "role": "user",
                    "content": f"What color is the {shape} in the image?"
                },
                {
                    "role": "assistant",
                    "content": f"The {shape} is {color}."
                }
            ]
        })
        
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"Synthetic dataset generated successfully at {DATA_PATH}!")

class VisionQADataset(Dataset):
    def __init__(self, filepath, processor):
        self.processor = processor
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format chat template for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": conversations[0]["content"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": conversations[1]["content"]}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        
        # Apply processor
        processed = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in processed.items()}
        # For causal LM vision models, labels are identical to input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs

def train_vision():
    # 1. Generate Dataset
    if not os.path.exists(DATA_PATH):
        generate_synthetic_vision_dataset()
        
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    
    print("Configuring QLoRA 4-bit parameters...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print("Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 2. LoRA Config targeting SmolVLM parameters
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 3. Load Dataset
    dataset = VisionQADataset(DATA_PATH, processor)
    
    # 4. Training Arguments optimized for 8GB VRAM
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        max_steps=100, # Optimized to complete in under 1 hour
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        save_total_limit=1
    )
    
    def collate_fn(batch):
        # Dynamically pad batch elements to maximum sequence length
        input_ids = [item["input_ids"] for item in batch]
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequence inputs
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # Ignore loss on padding
        
        # Stack vision inputs
        stacked_pixel_values = torch.stack(pixel_values)
        
        # Build batch dictionary
        batch_dict = {
            "input_ids": padded_input_ids,
            "pixel_values": stacked_pixel_values,
            "labels": padded_labels
        }
        
        # If model expects attention mask
        if "attention_mask" in batch[0]:
            attention_masks = [item["attention_mask"] for item in batch]
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            batch_dict["attention_mask"] = padded_attention_masks
            
        return batch_dict

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )
    
    print("Starting Vision Model training...")
    trainer.train()
    
    print("Saving Vision adapters...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"[SUCCESS] Vision Model trained and adapters saved at {OUTPUT_DIR}!")

if __name__ == "__main__":
    train_vision()
