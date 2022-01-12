#!/bin/sh
./quantize.py trained/simplesort5_qat.pth.tar trained/simplesort5_qat-q.pth.tar --device MAX78000 -v "$@"
