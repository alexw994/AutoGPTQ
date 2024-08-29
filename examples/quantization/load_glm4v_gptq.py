import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import json
import random
import time

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

device = 'cuda:0'
quantized_model_dir = 'alexwww94/glm-4v-9b-gptq'
trust_remote_code = True

tokenizer = AutoTokenizer.from_pretrained(
    quantized_model_dir,
    trust_remote_code=trust_remote_code,
)

# model = AutoGPTQForCausalLM.from_quantized(
#     quantized_model_dir,
#     device=device,
#     trust_remote_code=trust_remote_code,
#     torch_dtype=torch.float16,
#     # use_cache=True
#     # use_marlin=True
# )

model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=trust_remote_code,
).to(device)

dataset = datasets.load_dataset('alexwww94/CogVLM-SFT-311K-subset-gptq')

for example in dataset['single']:
    # prompt = "为什么马会被围栏限制在一个区域内？"
    prompt = json.loads(example['labels_zh'])['conversations'][0]
    answer = json.loads(example['labels_zh'])['conversations'][1]
    image = example['image']
    print(f"prompt: {prompt['content']}")
    print("-" * 42)
    print(f"golden: {answer['content']}")
    print("-" * 42)

    start = time.time()

    prompt.update({'image': image})
    inputs = tokenizer.apply_chat_template([prompt],
                                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                    return_dict=True, dtyp=torch.bfloat16)  # chat mode
    
    inputs = inputs.to(device)
    inputs['images'] = inputs['images'].half()
    model = model.to(device).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(outputs[0]).split('<|endoftext|>')[0]

    end = time.time()
    print(f"quant: {generated_text}")
    num_new_tokens = len(tokenizer(generated_text)["input_ids"])
    print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
    print("=" * 42)

    break
