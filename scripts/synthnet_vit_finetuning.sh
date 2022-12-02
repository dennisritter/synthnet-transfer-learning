#!/bin/bash
PROJECT_NAME='synthnet-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_finetuning/vit'

TRAIN_DS='data/topex-printer/train'
TEST_DS='data/topex-printer/test'
SPLIT_VAL_SIZE=0.1

MODE='FINETUNING'

EPOCHS=3

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--split_val_size $SPLIT_VAL_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS