import json
import os
import pandas as pd
import torch
import re
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-1.1-7b-it"
ADAPTER_PATH = "./lora-traffic-reasoning"
YOLO_CACHE = "yolo_tracking_cache.json"
TEST_JSON = "test.json"  # Ensure this path matches your directory structure
SUBMISSION_CSV = "submission.csv"

def extract_numbers_for_time(text):
    """Attempt to find two numbers in the text for start/end times."""
    numbers = re.findall(r"\d+\.?\d*", str(text))
    if len(numbers) >= 2:
        return {"start": float(numbers[0]), "end": float(numbers[1])}
    return {"start": 0.0, "end": 1.0}

# --- 1. Load Existing Submissions (Resume Capability) ---
existing_indices = set()
submissions = []
if os.path.exists(SUBMISSION_CSV):
    print(f"Found existing {SUBMISSION_CSV}, loading completed predictions...")
    try:
        df_existing = pd.read_csv(SUBMISSION_CSV)
        existing_indices = set(df_existing["item_index"].tolist())
        submissions = df_existing.to_dict("records")
        print(f"Loaded {len(existing_indices)} existing predictions.")
    except Exception as e:
        print(f"Could not load existing CSV: {e}. Starting fresh.")

print("Loading model and adapter...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

with open(YOLO_CACHE, "r") as f:
    tracking_cache = json.load(f)

with open(TEST_JSON, "r") as f:
    test_data = json.load(f)

print("Running inference...")
for item in test_data["items"]:
    item_index = item["item_index"]
    
    # --- 2. Skip if already predicted ---
    if item_index in existing_indices:
        continue
        
    video_id = item["video_id"]
    yolo_context = tracking_cache.get(video_id, "No movement detected.")
    task_type = item.get("task_type", "")
    
    # --- 3. Add Format Hints to the Prompt ---
    format_hint = ""
    if task_type == "bcq":
        format_hint = " (Answer strictly with 'Yes' or 'No')"
    elif task_type == "mcq":
        format_hint = " (Answer strictly with the correct letter, e.g., A, B, C, or D)"
    elif task_type == "temporal_localization":
        format_hint = ' (Answer strictly in valid JSON format: {"start": [time], "end": [time]})'

    prompt = (
        f"User: You are a traffic anomaly expert. Based on the tracking data below, answer the question.{format_hint}\n"
        f"Tracking Data: {yolo_context}\n"
        f"Question: {item['question']}\n\n"
        f"Assistant: Reasoning:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the final answer format
    try:
        final_answer = response.split("Answer:")[-1].strip()
    except IndexError:
        final_answer = response.strip()
        
    # --- 4. Fallback logic to guarantee a parseable format on the fly ---
    if task_type == "bcq":
        if not re.search(r"\b(yes|no)\b", final_answer.lower()):
            final_answer = "Yes" 
            
    elif task_type == "mcq":
        if not re.search(r"\b([A-D])\b", final_answer):
            m = re.search(r"\b([a-d])\b", final_answer)
            if m:
                final_answer = m.group(1).upper()
            else:
                final_answer = "A"
                
    elif task_type == "temporal_localization":
        if "{" not in final_answer or "start" not in final_answer or "end" not in final_answer:
            time_json = extract_numbers_for_time(final_answer)
            final_answer = json.dumps(time_json)
        
    submissions.append({"item_index": item_index, "prediction": final_answer})
    
    # --- 5. Save incrementally to prevent data loss on crash ---
    df = pd.DataFrame(submissions)
    df.to_csv(SUBMISSION_CSV, index=False, quoting=1)

print("All predictions completed and saved safely to submission.csv")