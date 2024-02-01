#!/bin/bash

for i in 21; do
  python imagenet_FI_bit_2.py \
      --model_name VGG16 \
      --bit_pos $i \
      > imagenet_error/error_bit_vgg_${i}.log 2>&1 &
done