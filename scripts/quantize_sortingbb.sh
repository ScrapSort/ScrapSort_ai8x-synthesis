#!/bin/sh
./quantize.py trained/sort_qat_bb_l1.pth.tar trained/sort_qat_bb_l1-q.pth.tar --device MAX78000 -v "$@"
