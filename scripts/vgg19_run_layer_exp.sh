#!/bin/bash

for i in {20..25}; do
  python imagenet_FI_layer_2.py \
      --model_name VGG19 \
      --layer_no $i \
      > imagenet_error/error_layer_vgg19_${i}.log 2>&1 &
done