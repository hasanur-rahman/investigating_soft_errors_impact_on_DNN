#!/bin/bash

for i in {30..31}; do
  python imagenet_FI_bit_2.py \
      --model_name VGG19 \
      --bit_pos $i \
      > imagenet_error/error_bit_vgg19_${i}.log 2>&1 &
done