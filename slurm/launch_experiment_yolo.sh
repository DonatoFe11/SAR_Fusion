#!/bin/bash
conda init
conda activate sarfusion
python main.py experiment $@ --parallel --yolo
