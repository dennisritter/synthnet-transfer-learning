#!/bin/bash
PROJECT_NAME='visda2017-vit-finetuning'
RUN_NAME='vitb16_gs_augmix'
OUTPUT_DIR='out/synthnet_finetuning/synthnet_finetuning/vit'

TRAIN_DS='data/visda2017/train'
# VAL_DS='data/visda2017/val'
TEST_DS='data/visda2017/test'

MODE='FINETUNING'
EPOCHS=20
BATCH_SIZE=16
LR=3e-5
WEIGHT_DECAY=0.1
WARM_UP_RATIO=0.1

python synthnet_vit_finetuning.py \
--project_name $PROJECT_NAME \
--run_name $RUN_NAME \
--output_dir $OUTPUT_DIR \
--train_ds $TRAIN_DS \
--test_ds $TEST_DS \
--batch_size $BATCH_SIZE \
--mode $MODE \
--num_train_epochs $EPOCHS \
--learning_rate $LR \
--weight_decay $WEIGHT_DECAY \
--warmup_ratio $WARM_UP_RATIO
--grayscale True
--augmix True