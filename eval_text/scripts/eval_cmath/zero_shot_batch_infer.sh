model_path="./checkpoints/multimath-7b-llava-v1.5"
answers_file="./eval_text/outputs/cmath/zeroshot_multimath-7b-llava-v1_5.json"

CHUNKS=8
for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python3 eval_text/infer.py \
        --model_path ${model_path} \
        --dataset cmath \
        --conv_mode dpsk \
        --prompt multimath \
        --answers_file ${answers_file} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

python3 eval_text/merge_pred.py --answers_file ${answers_file}
