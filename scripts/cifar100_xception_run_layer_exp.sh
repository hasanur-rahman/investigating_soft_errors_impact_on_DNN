#!/bin/bash

for i in {166..170}; do
  python cifar100_FI_layer_2.py \
      --model_name xception \
      --layer_no $i \
      > cifar100_error/error_layer_xception_${i}.log 2>&1 &
done