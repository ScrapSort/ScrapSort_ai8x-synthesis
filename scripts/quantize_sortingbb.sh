#!/bin/sh
./quantize.py trained/sort_qat_bb_l12.pth.tar trained/sort_qat_bb_l12-q.pth.tar --device MAX78000 -v "$@"
