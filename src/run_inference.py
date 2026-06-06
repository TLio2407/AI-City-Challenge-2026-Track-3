import json
import re
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-1.1-7b-it"
ADAPTER_PATH = "./lora-traffic-reasoning"
YOLO_CACHE = "yolo_tracking_cache.json"
TEST_JSON = "test/test.json"

OPEN_ENDED_TASKS = frozenset({
    "bcq_openended", "mcq_openended", "open_qa",
    "causal_linkage", "scene_description", "temporal_description",
    "video_summarization",
})


def extract_yesno(text):
    """Extract Yes/No from response for bcq tasks."""
    if not text or not str(text).strip():
        return "No"
    s = str(text).strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    return "No"


def extract_letter(text):
    """Extract letter (A-D) from response for mcq tasks."""
    if not text or not str(text).strip():
        return "A"
    s = str(text).strip()
    m = re.match(r"^\(?([A-Da-d])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)
    return "A"


def extract_json(text):
    """Extract JSON object from response for temporal_localization tasks."""
    if not text or not str(text).strip():
        return '{"start": "0:00", "end": "0:01"}'
    s = str(text).strip()
    # Try to find JSON in code blocks
    m = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "start" in obj and "end" in obj:
                return json.dumps(obj)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) \
                    and "start" in obj[0] and "end" in obj[0]:
                return json.dumps(obj[0])
        except json.JSONDecodeError:
            pass
    # Try plain JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "start" in obj and "end" in obj:
            return json.dumps(obj)
    except json.JSONDecodeError:
        pass
    return '{"start": "0:00", "end": "0:01"}'


def format_prediction(task_type, response):
    """Format the model response according to task type requirements."""
    if task_type == "bcq":
        return extract_yesno(response)
    elif task_type == "mcq":
        return extract_letter(response)
    elif task_type == "temporal_localization":
        return extract_json(response)
    else:
        # Open-ended tasks: return the full response
        return response.strip() if response else ""


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
    task_type = item.get("task_type", "open_qa")
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
    
    # Extract the final answer from the response
    try:
        final_answer = response.split("Assistant: Reasoning:")[-1].strip()
    except IndexError:
        final_answer = response.strip()
    
    # Format according to task type
    formatted_prediction = format_prediction(task_type, final_answer)
        
    
    submissions.append({"item_index": item_index, "prediction": formatted_prediction})

df = pd.DataFrame(submissions)
df.to_csv("submission.csv", index=False, quoting=1)
print("Saved submission.csv")