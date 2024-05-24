#!/bin/bash


#rte:accuracy, epoch
#mrpc: f1, epoch
#stsb: spearmanr, epoch
#cola: matthews_correlation, epoch
#sst2: accuracy, steps
#qnli: accuracy, steps
#qqp: f1, steps
#mnli: accuracy, steps

#set learning rate
# task_to_lr = {'rte': 3e-5,
#         'mrpc': 3e-5,
#         'stsb': 2e-5,
#         'sst2': 1e-5,
#         'cola': 2e-5,
#         'qqp': 2e-5,
#         'mnli': 1e-5,
#         'qnli': 2e-5,
#         }

export TASK_NAME=rte
export LR=3e-05

for MAX_TRAIN_SAMPLES in 200 500 1000
do
  for DATA_SEED in 111 222 333 444 555
  do
    CUDA_VISIBLE_DEVICES=1 python train_glue_LM_baseline.py \
        --model_name_or_path RoBERTa-base \
        --task_name $TASK_NAME \
        --output_dir result/RoBERTa-base/$TASK_NAME/$MAX_TRAIN_SAMPLES/$DATA_SEED/ \
        --num_train_epochs 20 \
        --learning_rate $LR \
        --per_device_train_batch_size 32 \
        --max_seq_length 128 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model accuracy \
        --train_as_val True \
        --max_train_samples $MAX_TRAIN_SAMPLES\
        --low_resource_data_seed $DATA_SEED \
        --load_best_model_at_end \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --do_predict False \
        --fp16 \
        "$@"
  done
done
