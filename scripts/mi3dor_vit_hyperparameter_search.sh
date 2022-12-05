#!/bin/bash
PROJECT_NAME='mi3dor-vit-hyperparameters'
# PROJECT_NAME='synthnet-vit-hyperparameters'
OUTPUT_DIR='out/synthnet_finetuning/mi3dor_hps/vit' # + RUN_NAME on runtime

TRAIN_DS='data/mi3dor/train'
TEST_DS='data/mi3dor/test'
MODE='HP_SEARCH'
EPOCHS=10

wandb login $WANDB_API_KEY

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--mode $MODE \
--num_train_epochs $EPOCHS \
--hps_run_count 10