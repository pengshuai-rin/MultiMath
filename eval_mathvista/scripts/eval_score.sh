output_file="pred_checkpoints_release.json"

python3 eval_mathvista/calculate_score.py \
    --output_dir ./eval_mathvista/outputs/ \
    --output_file ${output_file} \
    --score_file score_${output_file} \
    --gt_file ./playground/MathVista/data/testmini.json