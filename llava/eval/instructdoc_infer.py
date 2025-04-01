import sys
import os, os.path as osp
import argparse
import json
from tqdm import tqdm

# import packages
from PIL import Image
import re

import torch

# llava import
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
        model_base=args.model_base,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens
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

    results = []

    # get conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "llavar" in model_name.lower(): # support llavar models
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    for d in tqdm(data):
        qs = d['conversations'][0]['value'] + ' Use 1 to 3 words to answer.'

        conv = conv_templates[conv_mode].copy()

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if 'image' in d:
            images = [Image.open(d['image']).convert('RGB')]
        elif 'image_list' in d:
            images = [Image.open(x).convert('RGB') for x in d['image_list']]
        else:
            raise ValueError("No image found in the input json.")
        
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.device)
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if model_cfgs['temperature'] > 0 else False,
                temperature=model_cfgs['temperature'],
                top_p=model_cfgs['top_p'],
                num_beams=model_cfgs['num_beams'],
                max_new_tokens=model_cfgs['max_new_tokens'],
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        d['conversations'][1]['value'] = outputs

        results.append(d)
    
    with open(args.output_json, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("test_json", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("--model_path", type=str, default='liuhaotian/llava-v1.6-vicuna-7b')
    parser.add_argument("--cache_dir", type=str, default='./checkpoints')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()
    main(args)