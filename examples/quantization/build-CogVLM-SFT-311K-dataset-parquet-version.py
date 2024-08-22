import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import pandas as pd
import glob
import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict

data_path = '/root/.cache/huggingface/datasets/downloads/CogVLM-SFT-311K'

details = 'llava_details-minigpt4_3500_formate'
single = 'llava_instruction_single_conversation_formate'
multi = 'llava_instruction_multi_conversations_formate'


def gene(split):
    images = glob.glob(os.path.join(data_path, split, 'images', '*.jpg'))

    for i in images:
        image = Image.open(i)
        c_en = os.path.join(data_path, split, 'labels_en', os.path.basename(i).replace('.jpg', '.json'))
        c_zh = os.path.join(data_path, split, 'labels_zh', os.path.basename(i).replace('.jpg', '.json'))

        if not os.path.exists(c_en) or not os.path.exists(c_zh):
            continue

        with open(c_en, 'r') as f:
            c_en = json.dumps(json.load(f))

        with open(c_zh, 'r', encoding='utf8') as f:
            c_zh = json.dumps(json.load(f), ensure_ascii=False)

        yield {'image': image, 'labels_en': c_en, 
                    'labels_zh': c_zh, 'file_name': os.sep.join(os.path.abspath(i).split(os.sep)[-4:])}

dataset = DatasetDict(details=Dataset.from_generator(gene, gen_kwargs={"split": details}),
                    single=Dataset.from_generator(gene, gen_kwargs={"split": single}),
                    multi=Dataset.from_generator(gene, gen_kwargs={"split": multi}))

dataset.push_to_hub("alexwww94/CogVLM-SFT-1K")