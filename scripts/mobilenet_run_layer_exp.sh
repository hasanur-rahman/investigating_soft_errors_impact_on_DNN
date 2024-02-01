#!/bin/bash
# 3, 5, 6, 9, 12, 14, 22
for i in 12 14 22; do
  python imagenet_FI_layer_2.py \
      --model_name MobileNet \
      --layer_no $i \
      > imagenet_error/error_layer_xception_${i}.log 2>&1 &
done