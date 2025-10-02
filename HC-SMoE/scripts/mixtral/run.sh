export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29514 hcsmoe/merging-mixtral.py \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --model_name="/data/zhaoys/Mixtral-8x7B-v0.1" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hirarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=6 \
  --n_sentences=128 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --start_layer=0 \
  --result_path="results/result_mixtral_test.txt" \
  --output_path="results/"