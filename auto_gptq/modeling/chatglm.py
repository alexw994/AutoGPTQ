from logging import getLogger
from torch import LongTensor, BFloat16Tensor

from ._base import BaseGPTQForCausalLM, List, Dict, torch, Union
import copy
import logging
import os
from os.path import isdir, join
from typing import Dict, List, Optional, Union, Iterable

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate.hooks import remove_hook_from_module
from safetensors import safe_open
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import (
    CommitOperationAdd,
    PushToHubMixin,
    create_commit,
    create_repo,
)

from ..nn_modules._fused_base import FusedBaseAttentionModule, FusedBaseMLPModule
from ..nn_modules.qlinear import GeneralQuantLinear
from ..quantization import GPTQ, BaseQuantizeConfig
from ..quantization.config import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_FIELD,
    QUANT_METHOD_FIELD,
    QUANTIZE_BLACK_LIST,
)
from ..utils.accelerate_utils import load_checkpoint_in_model
from ..utils.data_utils import collate_data
from ..utils.import_utils import (
    AUTOGPTQ_CUDA_AVAILABLE,
    EXLLAMA_KERNELS_AVAILABLE,
    EXLLAMAV2_KERNELS_AVAILABLE,
    MARLIN_AVAILABLE,
    QIGEN_AVAILABLE,
    TRITON_AVAILABLE,
    dynamically_import_QuantLinear,
)
from ..utils.marlin_utils import (
    _validate_marlin_compatibility,
    _validate_marlin_device_support,
    prepare_model_for_marlin_load,
)
from ._const import CPU, CUDA_0, SUPPORTED_MODELS
from ._utils import (
    autogptq_post_init,
    find_layers,
    get_checkpoints,
    get_device,
    get_module_by_name_prefix,
    get_module_by_name_suffix,
    make_quant,
    make_sure_no_tensor_in_meta_device,
    move_to_device,
    pack_from_tensors,
    pack_model,
    preprocess_checkpoint_qigen,
    simple_dispatch_model,
    unpack_awq,
)
import copy

logger = getLogger(__name__)

def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v


def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [LongTensor(block["labels"]) for block in blocks]
    position_ids = [LongTensor(block["position_ids"]) for block in blocks]
    images = [BFloat16Tensor(block["images"]) for block in blocks]

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
            position_ids[i] = pad_block(position_ids[i], torch.zeros((block_bsz, pad_num)))

        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    return {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long(),
        "position_ids": torch.cat(position_ids, dim=0).long(),
        "images": torch.cat(images, dim=0),
    }



def after_lm_layer_quantized(layer, layer_inputs, layer_input_kwargs, layer_outputs, cur_layer_device, num_batches, cache_examples_on_gpu):
    for j in range(num_batches):
        layer_input = []
        for k, layer_inp in enumerate(layer_inputs[j]):
            layer_input.append(move_to_device(layer_inp, cur_layer_device))

        layer_output = []
        hidden_states, kv_cache = layer(*layer_input, **layer_input_kwargs[j])
        attention_mask = None
        rotary_pos_emb = layer_inputs[j][2]

        for k, layer_out in enumerate([hidden_states, attention_mask, rotary_pos_emb]):
            layer_output.append(move_to_device(layer_out,
            cur_layer_device if cache_examples_on_gpu else CPU,
        ))
        layer_outputs.append(layer_output)



def after_vm_layer_quantized(layer, layer_inputs, layer_input_kwargs, layer_outputs, cur_layer_device, num_batches, cache_examples_on_gpu):
    for j in range(num_batches):
        layer_input = []
        for k, layer_inp in enumerate(layer_inputs[j]):
            layer_input.append(move_to_device(layer_inp, cur_layer_device))

        layer_output = []
        hidden_states = layer(*layer_input, **layer_input_kwargs[j])

        for k, layer_out in enumerate([hidden_states]):
            layer_output.append(move_to_device(layer_out,
            cur_layer_device if cache_examples_on_gpu else CPU,
        ))
        layer_outputs.append(layer_output)


class ChatGLMGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = ["GLMBlock", "TransformerLayer", "GLU"]

    layers_block_names = ["transformer.encoder.layers", 
                            "transformer.vision.transformer.layers", 
                            "transformer.vision.linear_proj"]
        
    outside_layer_modules = ["transformer.embedding.word_embeddings", "transformer.output_layer", 
                             "transformer.vision.patch_embedding", "transformer.vision.conv"]
    
    inside_layer_modules = [
        ["self_attention.query_key_value", "self_attention.dense", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],

        # ============================================ #

        ["attention.query_key_value", "attention.dense", "mlp.fc1", "mlp.fc2"],

        # ============================================ #

        ["linear_proj", "dense_h_to_4h", "gate_proj", "dense_4h_to_h"],
    ]


    def _prepare_examples_for_quantization(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
    ):
        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_examples = []
        for example in examples:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])
            if "labels" in example:
                labels = _convert_tensor_to_list(example["labels"])
            elif "label" in example:
                labels = _convert_tensor_to_list(example["label"])
            elif "label_ids" in example:
                labels = _convert_tensor_to_list(example["label_ids"])
            else:
                labels = copy.deepcopy(input_ids)
            
            if "images" in example:
                images = example["images"].cpu().numpy().tolist()
            else:
                images = None

            if "position_ids" in example:
                position_ids = _convert_tensor_to_list(example["position_ids"])
            else:
                position_ids = None
            
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                } if position_ids is None or images is None else \
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "images": images,
                    "position_ids": position_ids
                }
            )

        pad_token_id = self.config.pad_token_id
        if not pad_token_id:
            pad_token_id = self.config.eos_token_id

        new_examples = [
            collate_data(new_examples[start : start + batch_size], pad_token_id)
            for start in range(0, len(new_examples), batch_size)
        ]
        for new_example in new_examples:
            del new_example["labels"]

        return new_examples

    def quantize_module(
        self,
        layers_block_name,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        cache_examples_on_gpu: bool = True,
        after_layer_quantized = None
    ):
        
        layer_inputs = []
        attention_masks = []
        position_ids = []
        images = []
        layer_input_kwargs = []
        layer_outputs = []

        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        num_batches = len(examples)
        layers = get_module_by_name_prefix(self.model, layers_block_name)

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if cache_examples_on_gpu else CPU
        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to_device(inp, data_device))
            layer_inputs.append(layer_input)

            one_kwargs = {}
            for (
                k,
                v,
            ) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to_device(v, data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

        force_layer_back_to_cpu = False
        if get_device(layers[0]) == CPU:
            layers[0] = layers[0].to(CUDA_0)
            force_layer_back_to_cpu = True

        ori_outside_layer_module_devices = {}
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to_device(module, cur_layer_device)

        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for example in examples:
            for k, v in example.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                example[k] = move_to_device(v, cur_layer_device)
            try:
                self.model(**example)
            except ValueError:
                pass
        handle.remove()

        move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
        for module_name in self.outside_layer_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to_device(module, ori_outside_layer_module_devices[module_name])

        torch.cuda.empty_cache()

        inside_layer_modules = self.inside_layer_modules
        if not self.quantize_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        quantizers = {}
        for i in range(len(layers)):
            logger.info(f"Start quantizing layer {i + 1}/{len(layers)}")
            layer = layers[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, CUDA_0)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = find_layers(layer)
            for names in inside_layer_modules:
                subset = {n: full[n] for n in names if n in full}
                logger.info(f"{i + 1}/{len(layers)} layer: {subset.keys()}")
                if len(subset) == 0:
                    continue
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        self.quantize_config.bits,
                        perchannel=True,
                        sym=self.quantize_config.sym,
                        mse=False,
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        # gptq is mutable.
                        gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(num_batches):
                    layer_input = []
                    for k, layer_inp in enumerate(layer_inputs[j]):
                        layer_input.append(move_to_device(layer_inp, cur_layer_device))

                    layer(*layer_input, **layer_input_kwargs[j])
                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)}...")
                    scale, zero, g_idx = gptq[name].fasterquant(
                        percdamp=self.quantize_config.damp_percent,
                        group_size=self.quantize_config.group_size,
                        actorder=self.quantize_config.desc_act,
                        static_groups=self.quantize_config.static_groups,
                    )
                    quantizers[f"{layers_block_name}.{i}.{name}"] = (
                        gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
                        move_to_device(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device),
                    )
                    gptq[name].free()

            after_layer_quantized(layer, layer_inputs, layer_input_kwargs, layer_outputs, cur_layer_device, num_batches, cache_examples_on_gpu)

            layers[i] = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []  # TODO: is it really OK to cache only the first positional argument?
            torch.cuda.empty_cache()

        return quantizers, force_layer_back_to_cpu


    @torch.inference_mode()
    def quantize(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
        batch_size: int = 1,
        use_triton: bool = False,
        use_cuda_fp16: bool = True,
        autotune_warmup_after_quantized: bool = False,
        cache_examples_on_gpu: bool = True,
    ):
        if self.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}")

        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        device_map = self.hf_device_map
        if device_map:
            for name, device in device_map.items():
                if device == "cpu":
                    logger.info(f"truly offloading {name} to cpu with hook.")
                    module = get_module_by_name_suffix(self.model, name)
                    remove_hook_from_module(module, recurse=True)
                    accelerate.cpu_offload_with_hook(module, CUDA_0)

        # examples = self._prepare_examples_for_quantization(examples, batch_size)
        examples = torch.load('/root/examples.data')

        # DONE
        quantizers_lm, force_layer_back_to_cpu_lm = self.quantize_module(self.layers_block_names[0],
                                                                examples, 
                                                                cache_examples_on_gpu,
                                                                after_layer_quantized=after_lm_layer_quantized)          

        # TODO
        quantizers_vm, force_layer_back_to_cpu_vm = self.quantize_module(self.layers_block_names[1],
                                                                examples, 
                                                                cache_examples_on_gpu,
                                                                after_layer_quantized=after_vm_layer_quantized)  

        quantizers_vm_glu, force_layer_back_to_cpu_vm = self.quantize_module(self.layers_block_names[2],
                                                                        examples, 
                                                                        cache_examples_on_gpu,
                                                                        after_layer_quantized=after_vm_layer_quantized)  

        quantizers_lm.update(quantizers_vm)
        quantizers_lm.update(quantizers_vm_glu)
        quantizers = quantizers_lm

        assert force_layer_back_to_cpu_lm == force_layer_back_to_cpu_vm
        force_layer_back_to_cpu = force_layer_back_to_cpu_lm

        forward_pass_use_cache = self.model.config.use_cache

        pack_model(
            model=self.model,
            quantizers=quantizers,
            bits=self.quantize_config.bits,
            group_size=self.quantize_config.group_size,
            use_triton=use_triton,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=self.quantize_config.desc_act,
            warmup_triton=autotune_warmup_after_quantized,
            force_layer_back_to_cpu=force_layer_back_to_cpu,
            use_marlin=self.quantize_config.checkpoint_format == CHECKPOINT_FORMAT.MARLIN,
        )
        if device_map:
            self.model = remove_hook_from_module(self.model, recurse=True)
            self.model = simple_dispatch_model(self.model, device_map)
        self.model.config.use_cache = forward_pass_use_cache

        self._quantized = True

        torch.cuda.empty_cache()

__all__ = ["ChatGLMGPTQForCausalLM"]
