#!/bin/bash

TRAIN_DS='/home/dennis/Desktop/work/evaluation_pipeline_data/modelnet10/train/images'
TEST_DS='/home/dennis/Desktop/work/evaluation_pipeline_data/modelnet10/test/images'
OUT='data/modelnet10'

python data_conversion.py \
--input_train_ds $TRAIN_DS \
--input_test_ds $TEST_DS \
--output_dir $OUT