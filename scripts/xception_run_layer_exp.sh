#!/bin/bash
# 15 56 65 75 83 84 90 103 107 118 120
for i in 75 83; do
  python imagenet_FI_layer_2.py \
      --model_name Xception \
      --layer_no $i \
      > imagenet_error/error_layer_xception_${i}.log 2>&1 &
done