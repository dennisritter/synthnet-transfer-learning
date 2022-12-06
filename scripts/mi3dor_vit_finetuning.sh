#!/bin/bash
PROJECT_NAME='mi3dor-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/mi3dor_finetuning/vit'

TRAIN_DS='data/mi3dor/train'
TEST_DS='data/mi3dor/test'
SPLIT_VAL_SIZE=0.1

MODE='FINETUNING'
EPOCHS=50
BATCH_SIZE=16
LR=3e-5
WEIGHT_DECAY=0.09
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
# --weight_decay $WEIGHT_DECAY 


# atomic-voice-1 // 87b93r8m :: 20 epochs
# CONTINUATION

EPOCHS=50
python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--split_val_size $SPLIT_VAL_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
#--resume True \
#--resume_id ABC123