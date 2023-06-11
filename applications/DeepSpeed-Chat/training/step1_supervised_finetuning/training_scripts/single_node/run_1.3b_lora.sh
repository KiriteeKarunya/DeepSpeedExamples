#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output/opt13b
mkdir -p $OUTPUT_PATH

deepspeed main.py --data_path Dahoas/synthetic-instruct-gptj-pairwise KiriteeGak/boat-data \
   --data_split 6,2,2 \
   --model_name_or_path facebook/opt-13b \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 3 \
   --lora_dim 32 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT_PATH
   # &> $OUTPUT_PATH/training.log
