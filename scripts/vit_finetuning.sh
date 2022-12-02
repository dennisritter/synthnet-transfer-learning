#!/bin/bash
PROJECT_NAME='synthnet-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/vit' # + RUN_NAME on runtime

TRAIN_DS='data/7054-12-300-l_drucker_se_su_st_st_512_32' # todo
TEST_DS='data/topex-real-123_pb_256' # todo
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