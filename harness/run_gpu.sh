#!/usr/bin/env bash
# 12-hour single-GPU experiment plan for the YB-Mixer harness.
# Edit/comment phases to fit your budget. Each run writes runs/<tag>.{csv,json}.
# Rough times assume one modern GPU (A100/3090-class) with --amp.
set -e
cd "$(dirname "$0")"
mkdir -p runs
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
COMMON="--amp --workers 4 --out runs"

echo "######## Phase 1: permuted-MNIST  (real data sanity, ~1-1.5h total) ########"
for M in yb transformer s4d mamba; do
  python train.py --task pmnist --model $M --dim 128 --depth 4 --epochs 40 \
                  --batch_size 64 --lr 3e-3 $COMMON
done

echo "######## Phase 2: sequential-CIFAR == LRA-Image  (flagship, ~6-7h total) ########"
for M in yb s4dlin lru fnet transformer mamba; do   # s4dlin = fair HiPPO SSM; add seeds below
  python train.py --task scifar --model $M --dim 128 --depth 6 --epochs 80 \
                  --batch_size 64 --lr 3e-3 --dropout 0.1 $COMMON
done
# a second seed for our model (variance):
python train.py --task scifar --model yb --dim 128 --depth 6 --epochs 80 \
                --batch_size 64 --lr 3e-3 --dropout 0.1 --seed 1 $COMMON

echo "######## Phase 3: byte-level IMDB ~ LRA-Text  (if time remains, ~3h) ########"
for M in yb transformer s4d; do
  python train.py --task imdb --model $M --dim 128 --depth 4 --epochs 25 \
                  --seq_len 1024 --batch_size 32 --lr 2e-3 $COMMON
done

echo "######## Results ########"
python aggregate.py runs
