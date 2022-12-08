#!/bin/bash
PROJECT_NAME='modelnet10-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/modelnet10_finetuning/vit'

TRAIN_DS='data/modelnet10/train'
TEST_DS='data/modelnet10/test'
SPLIT_VAL_SIZE=0.1

MODE='FINETUNING'
EPOCHS=20
BATCH_SIZE=64
LR=4e-5
WEIGHT_DECAY=0.05

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
--weight_decay $WEIGHT_DECAY 


# # atomic-voice-1 // 87b93r8m :: 20 epochs
# # CONTINUATION
# EPOCHS=20
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