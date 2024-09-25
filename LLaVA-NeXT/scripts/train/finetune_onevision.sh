export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

RANK=${RANK:-0}
ADDR=${ADDR:-"127.0.0.1"}
PORT=${PORT:-"29501"}
NNODES=${NNODES:-1}
NUM_GPUS=${NUM_GPUS:-1}

MID_RUN_NAME = "midrun.ckpt"

# LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION="/root/autodl-tmp/model/LLM/vicuna-7b-v1.5" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
# VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION="/root/autodl-tmp/model/visiontower/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME = "/root/model/MLLM/llava-v1.5-7b"

# BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint


 #--nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
ACCELERATE_CPU_AFFINITY=1 torchrun llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path "/root/autodl-tmp/MLLM/LLaVA-NeXT/playground/filtered_new_edit_data/img_diff_object_replacement.json" \
    --image_folder "/root/autodl-tmp/dataset/Img-Diff/object_replacement/filtered_new_edit_data" \
    --pretrain_mm_mlp_adapter="/root/autodl-tmp/model/MLLM/llava-v1.5-7b/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir "/checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32



    # --data_path "/root/autodl-tmp/MLLM/LLaVA/playground/data/img_diff copy.json" \
    # --data_path ./onevision_data.yaml \
    # --image_folder ./onevision_data/images \
    # --video_folder ./onevision_data/videos \
    # --pretrain_mm_mlp_adapter="/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    # --run_name $MID_RUN_NAME \
# You can delete the sdpa attn_implementation if you want to use flash attn