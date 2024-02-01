#!/bin/bash

for i in {0..31}; do
  python cifar100_FI_bit_2.py \
      --model_name resnet50 \
      --bit_pos $i \
      > cifar100_error/error_bit_resnet50_${i}.log 2>&1 &
done