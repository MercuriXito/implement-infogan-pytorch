#!/bin/bash

python -m utils.eval \
    --model-path outputs/infogan_mnist/netG.pt \
    --batch-size 32 \
    --data-name MNIST

