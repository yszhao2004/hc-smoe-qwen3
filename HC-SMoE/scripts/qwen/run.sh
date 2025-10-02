export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES=0

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-qwen.py \
  --model_name="/data/zhaoys/qwen-moe" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hirarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=45 \
  --n_sentences=128 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/merge-30e/test"