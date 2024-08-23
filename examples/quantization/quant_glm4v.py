import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
# from vllm import LLM
# LLM("THUDM/glm-4v-9b", trust_remote_code=True, dtype='half')
# exit()

import json
import random
import time
import glob

from argparse import ArgumentParser
from datasets import Dataset, DatasetDict, load_dataset, Array4D, Features

import torch
from datasets import Dataset, DatasetDict
import datasets
from transformers import AutoTokenizer, TextGenerationPipeline

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import concatenate_datasets


def text_generator(model, tokenizer, prompt, device):
    inputs = tokenizer.apply_chat_template(prompt,
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode
    inputs = inputs.to(device)
    model = model.to(device)
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0]).split('<|endoftext|>')[0]


def load_data(data_path, tokenizer, n_samples):
    dataset = datasets.load_dataset(data_path)

    def tokenize(examples):
        if 'labels_zh' in examples:
            is_zh = True
            labels = examples['labels_zh']
        else:
            is_zh = False
            labels = examples['labels_en']

        prompts = []
        input_ids = []
        position_ids = []
        images = []
        attention_mask = []
        outputs = []
        for img, label in zip(examples['image'], labels):
            label = json.loads(label)

            prompt = label['conversations'][0:1]
            prompt[0]['image'] = img

            output = label['conversations'][1:2]
                
            tokenized_data = tokenizer.apply_chat_template(prompt,
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            position_ids.append(tokenized_data["position_ids"][: tokenizer.model_max_length])
            images.append(tokenized_data["images"])
            prompts.append(prompt)
            outputs.append(output)

        return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt": prompts,
            "output": outputs,
            "images": images}

    tic = time.time()    
    rst = []
    for split in ['single', 'multi']:
        for label in ['labels_en', 'labels_zh']:
            d = dataset[split].select_columns(['image', label])[:n_samples]
            d = tokenize(d)
            n = len(d['input_ids'])
            rst.extend([{k: d[k][i] for k in d.keys()} for i in range(n)])
    print(f"slice dataset: {1000 * (time.time() - tic)} ms")  

    return rst


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group size, -1 means no grouping or full rank",
    )
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )
    parser.add_argument(
        "--save_and_reload",
        action="store_true",
        help="whether save quantized model to disk and reload back",
    )
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="whether use triton to speedup at inference",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="max memory used to load model per gpu",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=int,
        default=None,
        help="max memory used to offload model to cpu",
    )
    parser.add_argument(
        "--quant_batch_size",
        type=int,
        default=1,
        help="examples batch size for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="whether to trust remote code when loading model",
    )
    args = parser.parse_args()

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    
    examples = load_data('alexwww94/CogVLM-SFT-311K-subset-gptq', tokenizer, args.num_samples)
    examples_for_quant = [
        {"input_ids": example["input_ids"], 
        "attention_mask": example["attention_mask"],
        "images": example["images"],
        "position_ids": example["position_ids"]} for example in examples
    ]

    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16
    )

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if not args.quantized_model_dir:
        args.quantized_model_dir = args.pretrained_model_dir

    if args.save_and_reload:
        model.save_quantized(args.quantized_model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            args.quantized_model_dir,
            device="cuda:0",
            use_triton=args.use_triton,
            inject_fused_mlp=True,
            inject_fused_attention=True,
            trust_remote_code=args.trust_remote_code,
        )

    generator_kwargs = {"model": model, "tokenizer": tokenizer, "device":'cuda:0'}

    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt'][0]['content']}")
        print("-" * 42)
        print(f"golden: {example['output'][0]['content']}")
        print("-" * 42)
        start = time.time()
        generated_text = text_generator(prompt=example['prompt'], **generator_kwargs)
        end = time.time()
        print(f"quant: {generated_text}")
        num_new_tokens = len(tokenizer(generated_text)["input_ids"])
        print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
        print("=" * 42)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
