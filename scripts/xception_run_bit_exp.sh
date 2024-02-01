#!/bin/bash

for i in 2 19; do
  python imagenet_FI_bit_2.py \
      --model_name Xception \
      --bit_pos $i \
      > imagenet_error/error_bit_xception_${i}.log 2>&1 &
done