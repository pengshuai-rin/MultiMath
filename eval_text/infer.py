import argparse
import json
import math
import os
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, evalmodel
from llava.utils import disable_torch_init

from config import dataset_config
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from few_shot_prompts import *

prompt_choice_zh = {
    "none": "",
    "deepseekmath": "\n请通过逐步推理来解答问题，并把最终答案放置于\boxed{}中。",
    "multimath": "\n请逐步推理并解答以下数学问题，并将最终答案放置于\\boxed{}中。\n每个步骤一行，使用如下形式：\nStep X (所使用的数学定理/依据): 具体解答步骤。\nAnswer: \\boxed{}"
}
prompt_choice_en = {
    "none": "",
    "deepseekmath": "\nPlease reason step by step, and put your final answer within \boxed{}.",
    "multimath": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nEach step is placed on a new line, using the following format: \nStep X (Mathematical theorem/basis used): Detailed solution steps. \nAnswer: \\boxed{}"
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_dataset(dataset, part='test_path'):
    json_path = dataset_config[dataset][part]
    questions = []
    for line in open(json_path).readlines():
        questions.append(json.loads(line.strip()))
    return questions

def process(que, dataset, prompt):
    if dataset == 'gsm8k':
        item = process_gsm8k_test(que)
        prompt_choice = prompt_choice_en
    elif dataset == 'math':
        item = process_math_test(que)
        prompt_choice = prompt_choice_en
    elif dataset == 'cmath':
        item = process_cmath(que)
        prompt_choice = prompt_choice_zh
    elif dataset == 'mgsm_zh':
        item = process_mgsm_zh(que)
        prompt_choice = prompt_choice_zh
    elif dataset == 'gaokao-mathcloze':
        item = process_agieval_gaokao_math_cloze(que)
        prompt_choice = prompt_choice_zh
    elif dataset == 'gaokao-mathqa':
        item = process_agieval_gaokao_mathqa(que)
        prompt_choice = prompt_choice_zh
    
    item['messages'][0]['content'] += prompt_choice[prompt]

    return item

def format_fewshot_prompt(item, dataset):
    if dataset == 'gsm8k':
        prompting = CoTGSMPrompt()
    elif dataset == 'math':
        prompting = MinervaMathPrompt()
    elif dataset == 'cmath':
        prompting = CoTCMATHPrompt()
    elif dataset == 'mgsm_zh':
        pass
    elif dataset == 'gaokao-mathcloze':
        dataset = CoTGaoKaoMathClozePrompt()
    elif dataset == 'gaokao-mathqa':
        dataset = CoTGaoKaoMathQAPrompt()

    return prompting.format_prompt(item['messages'][0]['content'], '')

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
    questions = load_dataset(args.dataset, args.part)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answers_file_dir = os.path.dirname(answers_file)
    os.makedirs(answers_file_dir, exist_ok=True)
    answers_file = os.path.join(answers_file_dir, f"{args.chunk_idx}_{os.path.basename(answers_file)}")
    ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        item = process(line, args.dataset, args.prompt)

        if args.few_shot == True:
            query = format_fewshot_prompt(item, args.dataset)
        else:
            query = item['messages'][0]['content'].strip()
        
        # fake image
        image_path = "./fake_image_336.png"

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
        response = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)
        
        if args.dataset == 'math':
            extract_answer_fn = extract_math_answer
        elif args.dataset == 'gaokao-mathcloze':
            extract_answer_fn = extract_agieval_gaokao_mathcloze_cot_test
        elif args.dataset == 'gaokao-mathqa':
            extract_answer_fn = extract_agieval_gaokao_mathqa_cot_test
        else:
            extract_answer_fn = extract_last_single_answer

        pred = extract_answer_fn(item['messages'][0]['content'], response, task='cot')

        item.update({
            'model_output': response,
            'prediction': pred,
        })

        ans_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices = ['gsm8k', 'math', 'cmath', 'mgsm_zh', 'gaokao-mathcloze', 'gaokao-mathqa'])
    parser.add_argument("--part", type=str, default='test_path')
    parser.add_argument("--conv_mode", type=str, default='dpsk')
    parser.add_argument("--prompt", type=str, choices = ['none', 'deepseekmath', 'multimath'])
    parser.add_argument("--few_shot", type=bool, default=False)
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
