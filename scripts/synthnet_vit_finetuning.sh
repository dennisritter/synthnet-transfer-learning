#!/bin/bash
PROJECT_NAME='synthnet-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_finetuning/vit'

TRAIN_DS='data/topex-printer/train'
# VAL_DS='data/topex-printer/test'
TEST_DS='data/topex-printer/test'

MODE='FINETUNING'
EPOCHS=50
BATCH_SIZE=8
LR=6e-5
WEIGHT_DECAY=0.01
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
# --val_ds $VAL_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO


# atomic-voice-1 // 87b93r8m :: 20 epochs
# CONTINUATION

# EPOCHS=50
# python synthnet_vit_finetuning.py \
# --project_name $PROJECT_NAME \
# --output_dir $OUTPUT_DIR \
# --train_ds $TRAIN_DS \
# --test_ds $TEST_DS \
# --split_val_size $SPLIT_VAL_SIZE \
# --mode $MODE \
# --num_train_epochs $EPOCHS \
# --per_device_train_batch_size $BATCH_SIZE \
# --per_device_eval_batch_size $BATCH_SIZE \
# --learning_rate $LR \
# --weight_decay $WEIGHT_DECAY \
# --resume True \
# --resume_id 87b93r8m