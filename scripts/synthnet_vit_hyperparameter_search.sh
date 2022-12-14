#!/bin/bash
PROJECT_NAME='synthnet-vit-hyperparameters'
# PROJECT_NAME='synthnet-vit-hyperparameters'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_hps/vit' # + RUN_NAME on runtime

TRAIN_DS='data/topex-printer/train'
VAL_DS='data/topex-printer/test'
TEST_DS='data/topex-printer/test'
MODE='HP_SEARCH'
BATCH_SIZE=256
# TOTAL_STEPS=
# WARMUP_STEPS=

wandb login $WANDB_API_KEY

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--val_ds $VAL_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \