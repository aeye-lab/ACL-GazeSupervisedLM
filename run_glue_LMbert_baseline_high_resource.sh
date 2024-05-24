#!/bin/bash

#rte:accuracy, epoch
#mrpc: f1, epoch
#stsb: spearmanr, epoch
#cola: matthews_correlation, epoch

#BERT
#task_to_lr = {'rte': 2e-5,
# 				'mrpc': 3e-5,
# 				'stsb': 4e-5,
# 				'sst2': 2e-5,
# 				'cola': 2e-5,
# 				'qqp': 2e-5,
# 				'mnli': 2e-5,
# 				'qnli': 2e-5,
#}

#RoBERT
export TASK_NAME=rte
export LR=2e-05

CUDA_VISIBLE_DEVICES=5 python train_glue_LM_baseline.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --output_dir result/bert/$TASK_NAME/All/ \
    --num_train_epochs 10 \
    --learning_rate $LR \
    --per_device_train_batch_size 32 \
    --max_seq_length 128 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model accuracy \
    --train_as_val False \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 \
    "$@"
