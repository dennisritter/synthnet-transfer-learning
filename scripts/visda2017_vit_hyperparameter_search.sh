#!/bin/bash
PROJECT_NAME='visda2017-vit-hyperparameters'
# PROJECT_NAME='synthnet-vit-hyperparameters'
OUTPUT_DIR='out/synthnet_finetuning/visda2017_hps/vit' # + RUN_NAME on runtime

TRAIN_DS='data/visda2017/train'
VAL_DS='data/visda2017/val'
TEST_DS='data/visda2017/test'
MODE='HP_SEARCH'
BATCH_SIZE=256

wandb login $WANDB_API_KEY

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--val_ds $VAL_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \