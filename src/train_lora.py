import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
    
# 1. Paths & Configuration
MODEL_ID = "google/gemma-1.1-7b-it" # Alternative: "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./lora-traffic-generalist"
YOLO_CACHE = "yolo_tracking_cache.json"
ANNOTATIONS_DIR = "train_annotations"

# Explicitly list all 10 tasks required by the AI City Challenge
TRAIN_FILES = [
    "bcq.json", "bcq_openended.json", "mcq.json", "mcq_openended.json",
    "open_qa.json", "scene_description.json", "video_summarization.json",
    "temporal_localization.json", "causal_linkage.json", "temporal_description.json"
]

def prepare_multitask_dataset():
    """Reads all 10 task JSON files, parses formats, and aggregates into a flat HF Dataset"""
    if not os.path.exists(YOLO_CACHE):
        raise FileNotFoundError(f"Could not find {YOLO_CACHE}. Please run Phase 1 (extract_features.py) first.")

    with open(YOLO_CACHE, "r") as f:
        tracking_cache = json.load(f)

    formatted_data = {"text": []}
    total_loaded_items = 0
    
    for file_name in TRAIN_FILES:
        file_path = os.path.join(ANNOTATIONS_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: Skipping {file_name}, file not found in {ANNOTATIONS_DIR}")
            continue
            
        with open(file_path, "r") as f:
            task_envelope = json.load(f)
            
        task_type = task_envelope.get("metadata", {}).get("task", file_name.replace(".json", ""))
        print(f"Loading task: {task_type} ({len(task_envelope['items'])} items)...")
        
        for item in task_envelope["items"]:
            video_id = item["video_id"]
            question = item["question"]
            reasoning = item["reasoning"]
            answer = item["answer"]
            
            # Extract tracking descriptions cached by YOLOv11
            yolo_context = tracking_cache.get(video_id, "No movement or objects detected.")
            
            # Format answers cleanly if they arrive as dictionary objects (common in temporal localization)
            if isinstance(answer, dict):
                answer_str = json.dumps(answer)
            else:
                answer_str = str(answer)
                
            # Build an explicit instructional prompt
            # Adding the [Task Context] tag forces the model to realize what specific rules apply
            prompt = (
                f"User: Task Category: {task_type}\n"
                f"Analyze the following traffic surveillance data and answer the question.\n"
                f"Tracking Data: {yolo_context}\n"
                f"Question: {question}\n\n"
                f"Assistant: Reasoning: {reasoning}\n"
                f"Answer: {answer_str}"
            )
            formatted_data["text"].append(prompt)
            total_loaded_items += 1
            
    print(f"Successfully compiled multi-task dataset. Total samples: {total_loaded_items}")
    return Dataset.from_dict(formatted_data)

# 2. Run Dataset Aggregation
dataset = prepare_multitask_dataset()
dataset = dataset.shuffle(seed=42) # Extremely important for mixed task files

# 3. Model Tokenization and Quantization Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit precision configuration...")

# Define the 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # Gemma models work well with bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 4. LoRA Weight Initialization
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Define Consolidated Training Rules
# Because the aggregated dataset is large (~44,040 samples), we adjust steps/epochs.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=2000,                 # Increase if you have the compute time to execute a full epoch
    save_steps=400,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

# 6. Execute SFT Pipeline
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1536,            # Ensure sequence lengths fit tracking sequence lengths
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting Unified Multi-Task LoRA Fine-Tuning Loop...")
trainer.train()

# 7. Persist Training Outputs
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Generalist Adapter saved successfully to {OUTPUT_DIR}")