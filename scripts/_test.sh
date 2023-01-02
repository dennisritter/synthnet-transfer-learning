#!/bin/bash
DATASET="topex-printer"

PROJECT_NAME='test'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_finetuning/test'

TRAIN_DS="data/${DATASET}/train"
# VAL_DS='data/topex-printer/test'
TEST_DS="data/${DATASET}/test"

MODE='FINETUNING'
EPOCHS=1
BATCH_SIZE=8
LR=6e-5
WEIGHT_DECAY=0.01
WARM_UP_RATIO=0.1
WORKERS=4

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO \
--workers $WORKERS \
--grayscale False \
--augmix True