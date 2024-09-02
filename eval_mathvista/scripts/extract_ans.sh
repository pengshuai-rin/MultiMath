output_file="pred_checkpoints_release.json"

python3 eval_mathvista/extract_answer.py \
    --output_dir ./eval_mathvista/outputs/ \
    --output_file ${output_file} \
    --llm_engine gpt-3.5-turbo \
    --api_key YOUR_API_KEY &