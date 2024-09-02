import argparse
import json
import math
import os
from tqdm import tqdm

from eval.eval_script import *

def eval(args):
    if args.dataset == 'math':
        eval_fn = eval_math
    elif args.dataset == 'gaokao-mathcloze':
        eval_fn = eval_agieval_gaokao_math_cloze
    elif args.dataset == 'gaokao-mathqa':
        eval_fn = eval_agieval_gaokao_mathqa
    else:
        eval_fn = eval_last_single_answer

    items = [json.loads(line.strip()) for line in open(args.answers_file).readlines()]
    o = open(args.answers_file, 'w')
    for item in tqdm(items):
        label = eval_fn(item)
        item['accuracy'] = label
        o.write(json.dumps(item, ensure_ascii=False)+'\n')
    
    print("Calculating accuracy...")
    acc = 0
    for item in items:
        acc += item['accuracy']
    print("output acc = {:.5f}".format(acc / len(items) * 100), flush=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices = ['gsm8k', 'math', 'cmath', 'mgsm_zh', 'gaokao-mathcloze', 'gaokao-mathqa'])
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    args = parser.parse_args()

    eval(args)