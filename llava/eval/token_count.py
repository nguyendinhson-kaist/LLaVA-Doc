import sys
import os, os.path as osp
import argparse
import json
from tqdm import tqdm

# import packages
from PIL import Image
import re

import torch

# vdinstruct import
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates

def main(args):
    assert osp.exists(args.test_json), f"Test json not found: {args.test_json}"
    assert not osp.exists(args.output_json), f"Output json already exists: {args.output_json}"

    # Load source json
    with open(args.test_json, "r") as f:
        data = json.load(f)

    # load model
    disable_torch_init()

    model_cfgs = dict(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        model_base=args.model_base
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please install CUDA to run this script.")
    
    model_name = get_model_name_from_path(model_cfgs['model_path'])
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_cfgs['model_path'], 
        model_cfgs['model_base'],
        model_name, 
        cache_dir=model_cfgs['cache_dir'], 
        device=device
    )

    results = {}

    for d in tqdm(data):
        if 'image' in d:
            image_paths = [osp.join(args.image_folder, d['image'])]
        elif 'image_list' in d:
            image_paths = [osp.join(args.image_folder, x) for x in d['image_list']]
        else:
            raise ValueError("No image found in the input json.")

        image_paths = [i for i in image_paths if i not in results]
        if len(image_paths) == 0:
            continue
        
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        with torch.inference_mode():
            image_tokens = model.get_image_features(images_tensor, image_sizes=image_sizes)
            image_tokens_count = [x.shape[0] for x in image_tokens]

        results.update({
            k: v for k, v in zip(image_paths, image_tokens_count)
        })

    average_count = int(sum(results.values()) / len(results))
    results.update({
        'average_count': average_count
    })
    print(f"Average token count: {average_count}")
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("test_json", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("--model_path", type=str, default='liuhaotian/llava-v1.6-vicuna-7b')
    parser.add_argument("--cache_dir", type=str, default='./checkpoints')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default='')

    args = parser.parse_args()
    main(args)