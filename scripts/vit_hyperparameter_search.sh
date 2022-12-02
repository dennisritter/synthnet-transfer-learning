#!/bin/bash
PROJECT_NAME='synthnet-vit-hyperparameters'
# PROJECT_NAME='synthnet-vit-hyperparameters'
OUTPUT_DIR='out/synthnet_hyperparameter_search/vit' # + RUN_NAME on runtime

TRAIN_DS='data/7054-12-300-l_drucker_se_su_st_st_512_32'
VAL_DS='data/7054-12-300-l_drucker_se_su_st_st_512_32'
TEST_DS='data/topex-real-123_pb_256'

MODE='HP_SEARCH'

EPOCHS=3

python synthnet_vit_finetune.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--val_ds $VAL_DS \
--test_ds $TEST_DS \
--mode $MODE \
--num_train_epochs $EPOCHS