#!/bin/bash

TRAIN_DS='data/evaluation_pipeline_input/7054-12-300-l_drucker_se_su_st_st_512_32/images'
TEST_DS='data/evaluation_pipeline_input/topex-real-123_pb_256/images'
OUT='data/topex-printer'

python data_conversion.py \
--input_train_ds $TRAIN_DS \
--input_test_ds $TEST_DS \
--output_dir $OUT