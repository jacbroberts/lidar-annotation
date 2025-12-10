from transformers import pipeline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
dtype = torch.bfloat16

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=dtype, 
    device_map="auto"
)

print(pipe("[{object=car, x=20, y=1,z=0}, {object=person, x=25, y=2,z=0}]: A natural language expression for this scene is"))

