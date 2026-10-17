#!/usr/bin/env bash
set -euo pipefail


dev=${GEO_DEVICE:-0}

# output and store energy scores for visualization

geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect \
  --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect \
  --use_reg --lamda 0.1 --m_in -7 --m_out -2 --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect \
  --use_prop --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset twitch --mode detect \
  --use_prop --use_reg --lamda 0.1 --m_in -5 --m_out -1 --use_bn --device "$dev"

geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect \
  --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect \
  --use_reg --lamda 0.1 --m_in -9 --m_out -4 --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect \
  --use_prop --use_bn --device "$dev"
geo discuss --dis_type vis_energy --method gnnsafe --backbone gcn --dataset arxiv --mode detect \
  --use_prop --use_reg --lamda 0.1 --m_in -9 --m_out -2 --use_bn --device "$dev"
