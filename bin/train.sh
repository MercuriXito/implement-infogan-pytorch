#!/bin/bash
python infoGAN.py --epoch 50 --batch-size 128 --data-name MNIST \
    --save-path outputs/infogan_mnist
