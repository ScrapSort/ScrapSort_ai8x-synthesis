#!/bin/sh
./quantize.py trained/simplesort4_qat.pth.tar trained/simplesort4_qat-q.pth.tar --device MAX78000 -v "$@"
