#!/bin/bash
DATASET="visda2017"

PROJECT_NAME="${DATASET}_DA"
RUN_NAME="resnet_in1k_augmix_lp"
OUTPUT_DIR="out/synthnet_finetuning/${PROJECT_NAME}"

TRAIN_DS="data/${DATASET}/train"
VAL_DS="data/${DATASET}/val"
TEST_DS="data/${DATASET}/test"

EPOCHS=20
BATCH_SIZE=16
LR=1e-3
WEIGHT_DECAY=0.1
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--model "microsoft/resnet-50" \
--project_name $PROJECT_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--val_ds $VAL_DS \
--test_ds $TEST_DS \
--run_name $RUN_NAME \
--train_layers "CLASS_HEAD" \
--seed 42 \
--batch_size $BATCH_SIZE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO \
--workers 8 \
--augmix True