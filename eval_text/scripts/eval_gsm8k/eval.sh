answers_file="./eval_text/outputs/gsm8k/zeroshot_multimath-7b-llava-v1_5.json"

python3 eval_text/cal_score.py \
        --dataset gsm8k \
        --answers_file ${answers_file}