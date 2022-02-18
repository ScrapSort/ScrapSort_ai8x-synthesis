#!/bin/sh
./quantize.py trained/simplesort7_qat.pth.tar trained/simplesort7_qat-q.pth.tar --device MAX78000 -v "$@"
