import fire
import re
from tqdm import tqdm
import time
import json
from utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os
import argparse


def evaluate(answer_file, regen_answer=False):
    lines = load_jsonl(answer_file)
    for line in tqdm(lines, desc='gen_correct'):
        raw_exampe = id_raw[line['id']]

        gt_answer = str(raw_exampe['answer']).strip()
        gt_answer_list = gt_answer.split('\n')
        if len(gt_answer_list) > 1:
            line['correct'] = False
            continue
        if len(raw_exampe['options']) > 0:
            gt_answer_value = raw_exampe['options'][gt_answer]
        else:
            gt_answer_value = gt_answer

        if 'model_answer' not in line or regen_answer:
            if line['response'] == None:
                model_answer == ''
            else:
                model_answer = line['response'].strip()
            for c in 'ABCDE':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            if 'oxed{' not in model_answer:
                # for flag in ['Answer:', 'the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                for flag in ['Answer:', 'the answer is']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                
            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').lstrip('option').strip()
            line['model_answer'] = model_answer
        else:
            model_answer = line['model_answer']
        line['correct'] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    save_jsonl(answer_file, lines, t_stamp=False)


def math_level_subject_acc(answer_file):
    print(answer_file)
    lines = load_jsonl(answer_file)
    
    results_dict = {}
    for line in tqdm(lines, desc='math_level_subject_acc'):
        correct = line['correct']
        raw_exampe = id_raw[line['id']]
        type = raw_exampe['problem_version']
        
        for key in [
            '-all', 
            f'-{type}'
            ]:
            if key not in results_dict:
                results_dict[key] = [0,0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1


    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'


    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(answer_file), ':\t', results_dict['-all'])
    json.dump(results_dict, open(answer_file.replace('.json', '_result.json'), 'w'), indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", type=str, default="/mnt/bn/pengshuai-nas/data/MathDataset/MathVerse/testmini.json")
    parser.add_argument("--answers_file", type=str, default="output.jsonl")
    args = parser.parse_args()

    id_raw = {example['id']: example for example in json.load(open(args.gt_file))}

    evaluate(args.answers_file, True)
    math_level_subject_acc(args.answers_file)
