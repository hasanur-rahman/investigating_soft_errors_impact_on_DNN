#!/bin/bash

for i in {28..31}; do
  python imagenet_FI_bit_2.py \
      --model_name MobileNet \
      --bit_pos $i \
      > imagenet_error/error_bit_mobilenet_${i}.log 2>&1 &
done