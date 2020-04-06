#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python3.6 -u main.py --epoch 150 --admm-quant --quant-type random_binary --resume "./checkpoints/cifar10_vgg16_bn_acc_93.92.pt" -a vgg16_bn --verbose
