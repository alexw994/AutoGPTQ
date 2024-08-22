from logging import getLogger
from torch import LongTensor

from ._base import BaseGPTQForCausalLM, List, Dict, torch, Union
import copy

logger = getLogger(__name__)


def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [LongTensor(block["labels"]) for block in blocks]
    position_ids = [LongTensor(block["position_ids"]) for block in blocks]
    image = [LongTensor(block["image"]) for block in blocks]

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
            image[i] = pad_block(image[i], torch.zeros((block_bsz, pad_num)))

        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    return {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long(),
        "position_ids": torch.cat(position_ids, dim=0).long(),
        "image": torch.cat(image, dim=0).long(),
    }


class ChatGLMGPTQForCausalLM(BaseGPTQForCausalLM):
    base_modules = ["transformer.embedding.word_embeddings", "transformer.output_layer"]

    layers_node = "transformer.encoder.layers"
    layer_type = "GLMBlock"
    layer_modules = [
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"],
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
            
            if "image" in example:
                image = _convert_tensor_to_list(example["image"])
            else:
                image = None

            if "position_ids" in example:
                position_ids = _convert_tensor_to_list(example["position_ids"])
            else:
                position_ids = None
            
            new_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                } if position_ids is None or image is None else \
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "image": image,
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


__all__ = ["ChatGLMGPTQForCausalLM"]
