#!/bin/bash
# 7 13 39 61 74 101 113 114 115 116 117 118 119 126 159
for i in 114 115 116 117 118 119 126 159; do
  python imagenet_FI_layer_2.py \
      --model_name ResNet50 \
      --layer_no $i \
      > imagenet_error/error_layer_resnet50_${i}.log 2>&1 &
done