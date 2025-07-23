set -x

save_path=./checkpoints/Qwen2.5-7B-dpo

train_files=data/UltraFeedback/sft.parquet
val_files=data/UltraFeedback/test.parquet

read -r -d '' training_commands <<EOF
    train_dpo.py \
   --save_path ${save_path} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain /home/work2/work/zhangxq/endorm/hub/Qwen2.5-7B \
   --bf16 \
   --max_epochs 3 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --recipe mean \
   --dataset ${train_files} \
   --eval_dataset ${val_files} \
   --eval_split train \
   --apply_chat_template \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --use_tensorboard ${save_path} \
   --eval_steps 15 \
   --full_determinism
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module --include localhost:0,1,2,3 $training_commands
fi
