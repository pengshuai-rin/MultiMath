import argparse
import json
import math
import os
import re
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, evalmodel
from llava.utils import disable_torch_init

from datasets import load_dataset

prompt_choice = {
    "none": "",
    "single": "Answer the question using a single word or phrase.",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def extract_boxed_content(text):
    pattern = r'\\boxed\{(.*?)\}'
    match_list = re.findall(pattern, text)
    if match_list == []:
        return None
    else:
        return match_list[-1]

def eval_model(args):
    # Model
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name
    )
    disable_torch_init()

    # Data
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answers_file_dir = os.path.dirname(answers_file)
    os.makedirs(answers_file_dir, exist_ok=True)
    answers_file = os.path.join(answers_file_dir, f"{args.chunk_idx}_{os.path.basename(answers_file)}")
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['query_cot']
        query = f"{question}\n"
        query += prompt_choice[args.prompt]
        if line['image'] == '':
            image_path = "/mnt/bn/pengshuai-nas/MathLLM/LLaVA/fake_image_336.png"
        else:
            image_path = os.path.join(args.image_folder, line["image"])

        args_llava = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": query,
            "conv_mode": args.conv_mode,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1024
        })()
        outputs = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)
        
        if args.prompt == 'multimath':
            response = extract_boxed_content(outputs)
        else:
            response = outputs
        
        line['model_output'] = outputs
        line['response'] = response
        ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--image_folder", type=str, default="/")
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    parser.add_argument("--prompt", type=str, choices=['none', 'single', 'multimath'])
    parser.add_argument("--conv_mode", type=str, default="dpsk")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    args = parser.parse_args()

    eval_model(args)
