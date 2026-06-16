import os
import json
import random

DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# System Prompt detailing the Zenthi-AI personality
SYSTEM_PROMPT = (
    "I am Zenthi-AI, a lightweight educational AI assistant developed as a custom Small Language Model project. "
    "I am educational, helpful, accurate, concise, student-friendly, and programming-aware."
)

def format_sharegpt_like(conversations):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    
    for turn in conversations:
        sender = turn.get("from") or turn.get("sender") or turn.get("role")
        val = turn.get("value") or turn.get("content")
        role = role_map.get(sender)
        if role and val:
            messages.append({"role": role, "content": val})
            
    # Need at least one user-assistant exchange
    if len(messages) >= 3:
        return {"messages": messages}
    return None

def load_json_or_jsonl(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                return json.loads(content)
            else:
                return [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as e:
        # Fallback to line-by-line streaming
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        return data

def process_alpaca(filepath):
    print("Processing Alpaca...")
    processed = []
    data = load_json_or_jsonl(filepath)
    for item in data:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        
        if not inst or not out:
            continue
            
        user_content = inst
        if inp:
            user_content += f"\nContext: {inp}"
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": out}
        ]
        processed.append({"messages": messages})
    return processed

def process_dolly(filepath):
    print("Processing Dolly 15K...")
    processed = []
    data = load_json_or_jsonl(filepath)
    for item in data:
        inst = item.get("instruction", "").strip()
        ctx = item.get("context", "").strip()
        resp = item.get("response", "").strip()
        
        if not inst or not resp:
            continue
            
        user_content = inst
        if ctx:
            user_content += f"\nContext: {ctx}"
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": resp}
        ]
        processed.append({"messages": messages})
    return processed

def process_openhermes(filepath):
    print("Processing OpenHermes...")
    processed = []
    data = load_json_or_jsonl(filepath)
    for item in data:
        conversations = item.get("conversations")
        if conversations:
            formatted = format_sharegpt_like(conversations)
            if formatted:
                processed.append(formatted)
        else:
            inst = item.get("instruction", "").strip()
            out = item.get("output", "").strip()
            if inst and out:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": out}
                ]
                processed.append({"messages": messages})
    return processed

def process_ultrachat(filepath):
    print("Processing UltraChat...")
    processed = []
    data = load_json_or_jsonl(filepath)
    for item in data:
        msgs = item.get("messages") or item.get("conversations")
        if msgs:
            formatted_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in msgs:
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                if role in ["user", "human"]:
                    formatted_msgs.append({"role": "user", "content": content})
                elif role in ["assistant", "gpt"]:
                    formatted_msgs.append({"role": "assistant", "content": content})
            if len(formatted_msgs) >= 3:
                processed.append({"messages": formatted_msgs})
    return processed

def process_sharegpt(filepath):
    print("Processing ShareGPT...")
    processed = []
    data = load_json_or_jsonl(filepath)
    for item in data:
        conversations = item.get("conversations")
        if conversations:
            formatted = format_sharegpt_like(conversations)
            if formatted:
                processed.append(formatted)
    return processed

def clean_and_dedup(datasets_list):
    print("Cleaning and deduplicating...")
    seen = set()
    cleaned = []
    
    for item in datasets_list:
        has_none = False
        user_parts = []
        for m in item["messages"]:
            if m.get("content") is None:
                has_none = True
                break
            if m["role"] == "user":
                user_parts.append(str(m["content"]))
                
        if has_none:
            continue
            
        user_msgs = " ".join(user_parts)
        if user_msgs not in seen:
            seen.add(user_msgs)
            cleaned.append(item)
            
    print(f"Removed {len(datasets_list) - len(cleaned)} duplicates/invalid items.")
    return cleaned

def generate_report(train_data, val_data, stats):
    report_path = os.path.join(OUTPUT_DIR, "dataset_report.json")
    report = {
        "statistics": stats,
        "train_samples": len(train_data),
        "val_samples": len(val_data)
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
    print(f"Report generated at {report_path}")

def main():
    alpaca = process_alpaca(os.path.join(DATASETS_DIR, "alpaca.json"))
    dolly = process_dolly(os.path.join(DATASETS_DIR, "dolly_15k.jsonl"))
    hermes = process_openhermes(os.path.join(DATASETS_DIR, "openhermes_subset.json"))
    ultrachat = process_ultrachat(os.path.join(DATASETS_DIR, "ultrachat_subset.json"))
    sharegpt = process_sharegpt(os.path.join(DATASETS_DIR, "sharegpt_subset.json"))
    
    all_data = alpaca + dolly + hermes + ultrachat + sharegpt
    print(f"Total raw examples merged: {len(all_data)}")
    
    cleaned = clean_and_dedup(all_data)
    print(f"Total cleaned examples: {len(cleaned)}")
    
    # Shuffle and split 90/10 train/validation
    random.seed(42)
    random.shuffle(cleaned)
    split_idx = int(len(cleaned) * 0.9)
    train_data = cleaned[:split_idx]
    val_data = cleaned[split_idx:]
    
    train_filepath = os.path.join(DATASETS_DIR, "train.json")
    val_filepath = os.path.join(DATASETS_DIR, "val.json")
    
    with open(train_filepath, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open(val_filepath, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
        
    print(f"Saved {len(train_data)} train samples to {train_filepath}")
    print(f"Saved {len(val_data)} validation samples to {val_filepath}")
    
    stats = {
        "alpaca_raw": len(alpaca),
        "dolly_raw": len(dolly),
        "openhermes_raw": len(hermes),
        "ultrachat_raw": len(ultrachat),
        "sharegpt_raw": len(sharegpt),
        "total_raw": len(all_data),
        "total_cleaned": len(cleaned)
    }
    generate_report(train_data, val_data, stats)

if __name__ == "__main__":
    main()
