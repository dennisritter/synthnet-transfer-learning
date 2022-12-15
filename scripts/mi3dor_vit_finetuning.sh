#!/bin/bash
PROJECT_NAME='mi3dor-vit-finetuning'
OUTPUT_DIR='out/synthnet_finetuning/mi3dor_finetuning/vit'

TRAIN_DS='data/mi3dor/train'
# VAL_DS='data/mi3dor/test'
TEST_DS='data/mi3dor/test'

MODE='FINETUNING'
EPOCHS=20
BATCH_SIZE=16
LR=3e-5
WEIGHT_DECAY=0.09
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
# --val_ds $VAL_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO