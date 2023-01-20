#!/bin/bash
DATASET="modelnet10meshgrid"

PROJECT_NAME="${DATASET}_DA"
RUN_NAME="vitb16_in21k_lpft"
OUTPUT_DIR="out/synthnet_finetuning/${PROJECT_NAME}"

TRAIN_DS="data/${DATASET}/train"
TEST_DS="data/${DATASET}/test"

EPOCHS=40
BATCH_SIZE=16
LR=3e-5
WEIGHT_DECAY=0.1
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--model "data/checkpoints/modelnet10meshgrid/vit-da-lp-step5000" \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--run_name $RUN_NAME \
--train_layers "FULL" \
--seed 42 \
--batch_size $BATCH_SIZE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO \
--workers 4 \
--augmix False
