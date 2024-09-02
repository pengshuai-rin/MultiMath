answers_file="./eval_text/outputs/gaokao-mathcloze/zeroshot_multimath-7b-llava-v1_5.json"

python3 eval_text/cal_score.py \
        --dataset gaokao-mathcloze \
        --answers_file ${answers_file}