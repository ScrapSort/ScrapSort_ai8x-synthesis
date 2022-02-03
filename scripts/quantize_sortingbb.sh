#!/bin/sh
./quantize.py trained/sort_qat_bb2.pth.tar trained/sort_qat_bb2-q.pth.tar --device MAX78000 -v "$@"
