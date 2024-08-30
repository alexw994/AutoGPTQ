import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import json
import random
import time

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

from auto_gptq.modeling._base import BaseGPTQForCausalLM
from auto_gptq.modeling.auto import GPTQ_CAUSAL_LM_MODEL_MAP

class ChatGLMGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = ["GLMBlock", "TransformerLayer", "GLU"]

    layers_block_names = ["transformer.encoder.layers", 
                            "transformer.vision.transformer.layers", 
                            "transformer.vision.linear_proj"]
        
    outside_layer_modules = ["transformer.output_layer"]
    
    inside_layer_modules = [
        ["self_attention.query_key_value", "self_attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        ["attention.query_key_value", "attention.dense", "mlp.fc1", "mlp.fc2"],
        ["linear_proj", "dense_h_to_4h", "gate_proj", "dense_4h_to_h"],
    ]

GPTQ_CAUSAL_LM_MODEL_MAP['chatglm'] = ChatGLMGPTQForCausalLM

device = 'cuda:0'
quantized_model_dir = 'alexwww94/glm-4v-9b-gptq-4bit'
trust_remote_code = True

tokenizer = AutoTokenizer.from_pretrained(
    quantized_model_dir,
    trust_remote_code=trust_remote_code,
)

model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir,
    device=device,
    trust_remote_code=trust_remote_code,
    torch_dtype=torch.float16,
    use_cache=True,
    inject_fused_mlp=True,
    inject_fused_attention=True,
)

# model = AutoModelForCausalLM.from_pretrained(
#     quantized_model_dir,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     trust_remote_code=trust_remote_code,
#     use_cache=True
# ).eval()

dataset = datasets.load_dataset('alexwww94/CogVLM-SFT-311K-subset-gptq')

for example in dataset['single']:
    # prompt = "为什么马会被围栏限制在一个区域内？"
    prompt = json.loads(example['labels_zh'])['conversations'][0]
    answer = json.loads(example['labels_zh'])['conversations'][1]
    image = example['image']
    print(f"prompt: {prompt}")
    print("-" * 42)
    print(f"golden: {answer}")
    print("-" * 42)

    start = time.time()

    prompt.update({'image': image})
    inputs = tokenizer.apply_chat_template([prompt],
                                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                    return_dict=True, dtyp=torch.bfloat16)  # chat mode
    inputs = inputs.to(device)
    inputs['images'] = inputs['images'].half()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(outputs[0]).split('<|endoftext|>')[0]

    end = time.time()
    print(f"quant: {generated_text}")
    num_new_tokens = len(tokenizer(generated_text)["input_ids"])
    print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
    print("=" * 42)

    # break
