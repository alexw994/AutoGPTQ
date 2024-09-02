import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 


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
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from typing import Any



def text_generator(model, tokenizer, prompt, device):
    inputs = tokenizer.apply_chat_template(prompt,
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode
    inputs = inputs.to(device)
    model = model.to(device)
    inputs['images'] = inputs['images'].half()
    
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0]).split('<|endoftext|>')[0]


class QuantDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, n_samples) -> None:
        super().__init__()
        ds = datasets.load_dataset(data_path)
        ds = concatenate_datasets([ds['single'], ds['multi']])
        self.ds = concatenate_datasets([ds.select_columns(['image', 'labels_en']).rename_column('labels_en', 'labels'),
                                        ds.select_columns(['image', 'labels_zh']).rename_column('labels_zh', 'labels')])

        self.tokenizer = tokenizer
        self.n_samples = n_samples
    

    def __getitem__(self, index) -> Any:
        example = self.ds[index]

        img = example['image']
        label = json.loads(example['labels'])
        prompt = label['conversations'][0]
        prompt['image'] = img

        tokenized_data = self.tokenizer.apply_chat_template([prompt],
                                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                    return_dict=True)

        input_ids = tokenized_data["input_ids"][: self.tokenizer.model_max_length]
        attention_mask = tokenized_data["attention_mask"][: self.tokenizer.model_max_length]
        position_ids = tokenized_data["position_ids"][: self.tokenizer.model_max_length]
        images = tokenized_data["images"]
        prompts = label['conversations'][0]
        outputs = label['conversations'][1]

        return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt": prompts,
            "output": outputs,
            "images": images}


    def __len__(self):
        return min(self.n_samples, len(self.ds))


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


    examples = QuantDataset('alexwww94/CogVLM-SFT-311K-subset-gptq', tokenizer, args.num_samples)

    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, 
                                           desc_act=args.desc_act, model_file_base_name='model.safetensors'),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code
    )

    start = time.time()
    model.quantize(
        examples,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton
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
