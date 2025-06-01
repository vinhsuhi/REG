
random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=8
STEP="0400000"
SAVE_PATH="your_path/linear-dinov2-b-enc8"
VAE_PATH="vae_path/"
NUM_STEP=250
MODEL_SIZE='L'
export WANDB_NAME=${SAVE_PATH}
export VAE_PATH=${VAE_PATH}
export NCCL_P2P_DISABLE=1




python ./evaluations/evaluator.py \
    --ref_batch your_path/VIRTUAL_imagenet256_labeled.npz \
    --sample_batch ${SAVE_PATH}/checkpoints/SiT-${MODEL_SIZE}-2-${STEP}-size-256-vae-ema-cfg-0.0-seed-0-sde-${NUM_STEP}.npz \
    --save_path ${SAVE_PATH}/checkpoints \
    --cfg_cond 0 \
    --step ${STEP} \
    --num_steps ${NUM_STEP}