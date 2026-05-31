import json
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-1.1-7b-it"
ADAPTER_PATH = "./lora-traffic-reasoning"
YOLO_CACHE = "yolo_tracking_cache.json"
TEST_JSON = "test/test.json"

print("Loading model and adapter...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

with open(YOLO_CACHE, "r") as f:
    tracking_cache = json.load(f)

with open(TEST_JSON, "r") as f:
    test_data = json.load(f)

submissions = []

print("Running inference...")
for item in test_data["items"]:
    item_index = item["item_index"]
    video_id = item["video_id"]
    yolo_context = tracking_cache.get(video_id, "No movement detected.")
    
    prompt = (
        f"User: You are a traffic anomaly expert. Based on the tracking data below, answer the question.\n"
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
        
    submissions.append({"item_index": item_index, "prediction": final_answer})

df = pd.DataFrame(submissions)
df.to_csv("submission.csv", index=False, quoting=1)
print("Saved submission.csv")