import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "meta-llama/Llama-3.2-1B"
dtype=torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", dtype=dtype)

model = PeftModel.from_pretrained(model, "./llama_scene_lora")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, dtype=dtype)

print(pipe("[{object=blender, x=3, y=1,z=4}, {object=person, x=4, y=3,z=1}]:",
           do_sample=True,
           temperature=0.1,
           top_p=0.9,
           max_new_tokens=100)[0]["generated_text"])