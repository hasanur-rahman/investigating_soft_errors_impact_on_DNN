#!/bin/bash

for i in 2 5; do
  python imagenet_FI_bit_2.py \
      --model_name ResNet50 \
      --bit_pos $i \
      > imagenet_error/error_bit_resnet50_${i}.log 2>&1 &
done