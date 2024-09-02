import argparse
import os
import json

def merge(args):
    base_name = os.path.basename(args.answers_file)
    src_dir = os.path.dirname(args.answers_file)
    tgt_dir = src_dir
    os.makedirs(tgt_dir, exist_ok=True)
    chunk_num = 8

    total_data = []
    for chunk_idx in range(chunk_num):
        json_path = os.path.join(src_dir, f"{chunk_idx}_{base_name}")
        chunk_data = open(json_path).readlines()
        print(chunk_idx, len(chunk_data))
        total_data.extend(chunk_data)
        os.remove(json_path)

    print(f"total {len(total_data)}")

    with open(os.path.join(tgt_dir, base_name), 'w') as o:
        for line in total_data:
            o.write(line.strip() + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str, default="/mnt/bn/pengshuai-nas/MathLLM/LLaVA/eval_text/outputs/pred_deepseekmathrl_mathstage1_llavastage2_math_stage3.json")
    args = parser.parse_args()

    merge(args)
