#!/usr/bin/env bash
# Reproduce every result in the paper. CPU-only; total runtime a few minutes.
set -e
cd "$(dirname "$0")/src"
for s in step1_gybe_primitive step2_learnable_gate step3_brickwall_stability \
         step4_transfer_matrix step5_trainable_transport step6_anytime_flow \
         step7_baselines_lengen step8_baselines step9_spectral_lengen; do
  echo; echo "########## $s ##########"
  python3 "$s.py"
done
