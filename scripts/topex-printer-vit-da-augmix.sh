#!/bin/bash
DATASET="topex-printer"

PROJECT_NAME="${DATASET}_DA"
RUN_NAME="vitb16_in21k_augmix"
OUTPUT_DIR="out/synthnet_finetuning/${PROJECT_NAME}"

TRAIN_DS="data/${DATASET}/train"
TEST_DS="data/${DATASET}/test"

EPOCHS=20
BATCH_SIZE=16
LR=3e-5
WEIGHT_DECAY=0.1
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--model "google/vit-base-patch16-224-in21k" \
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
--augmix True