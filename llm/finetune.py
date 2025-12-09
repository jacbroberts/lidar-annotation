from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("json", data_files="./llm/dataset.jsonl")

# Tokenization function
def tokenize(example):
    prompt = example["input"] + " " + example["target"]
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = dataset.map(tokenize)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    max_steps=500,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=200,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    data_collator=collator,
)

trainer.train()

# Save adapter
trainer.model.save_pretrained("./llama_scene_lora")
tokenizer.save_pretrained("./llama_scene_lora")