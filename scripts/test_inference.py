import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Zenthi-AI-merged"))
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Merged model not found at {model_path}")
        return
        
    print(f"Loading Zenthi-AI from local path: {model_path}...")
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # List of test prompts to evaluate identity and general instruction capability
    prompts = [
        "Who are you and who created you?",
        "What is Zenthi-AI?",
        "Write a quick Python function to check if a number is prime."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "="*50)
        print(f"Test Prompt {i}: {prompt}")
        print("="*50)
        
        messages = [
            {"role": "system", "content": "You are Zenthi-AI, a helpful, intelligent, and advanced AI assistant developed by KATHIR2006 (ELANGKATHIR11)."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Extract generated text (skipping the prompt)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Zenthi-AI response:\n{response}")
        except Exception as e:
            print(f"[ERROR] Generation failed for prompt {i}: {e}")

if __name__ == "__main__":
    main()
