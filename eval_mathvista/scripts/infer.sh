model_save_path="./checkpoints/multimath-7b-llava-v1.5"
output_file="pred_checkpoints_release.json"

python3 eval_mathvista/response.py \
    --rerun true \
    --conv_mode dpsk \
    --data_dir ./playground/MathVista/data \
    --input_file testmini.json \
    --output_dir ./eval_mathvista/outputs/ \
    --output_file  ${output_file} \
    --model_path ${model_save_path} &