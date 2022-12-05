#!/bin/bash
PROJECT_NAME='synthnet-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_finetuning/vit'

TRAIN_DS='data/topex-printer/train'
TEST_DS='data/topex-printer/test'
SPLIT_VAL_SIZE=0.1

MODE='FINETUNING'
EPOCHS=20
BATCH_SIZE=64
LR=6e-5
WEIGHT_DECAY=0.01

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