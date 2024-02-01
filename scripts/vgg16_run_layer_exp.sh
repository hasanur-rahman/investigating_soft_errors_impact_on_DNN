#!/bin/bash

for i in {19..21}; do
  python imagenet_FI_layer_2.py \
      --model_name VGG16 \
      --layer_no $i \
      > imagenet_error/error_layer_vgg_${i}.log 2>&1 &
done