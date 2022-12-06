#!/bin/bash
PROJECT_NAME='modelnet10-vit-hyperparameters'
# PROJECT_NAME='synthnet-vit-hyperparameters'
OUTPUT_DIR='out/synthnet_finetuning/modelnet10_hps/vit' # + RUN_NAME on runtime

TRAIN_DS='data/modelnet10/train'
TEST_DS='data/modelnet10/test'
MODE='HP_SEARCH'
EPOCHS=5

wandb login $WANDB_API_KEY

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--mode $MODE \
--num_train_epochs $EPOCHS \
--hps_run_count 10