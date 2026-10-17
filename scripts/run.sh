#!/usr/bin/env bash
set -euo pipefail

dev=${GEO_DEVICE:-0}
### Cora with structure ood

geo train --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn \
  --use_prop --use_reg --use_occ --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"

### Cora with feature ood

geo train --method msp --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn \
  --use_prop --use_reg --use_occ --m_in -5 --m_out -1 --lamda 0.01 --device "$dev"

### Cora with label ood

geo train --method msp --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --use_reg --m_in -5 --m_out -4 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --use_prop --use_reg --m_in -5 --m_out -4 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn \
  --use_prop --use_reg --use_occ --m_in -5 --m_out -4 --lamda 1.0 --device "$dev"


### Amazon-photo with structure ood

geo train --method msp --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --use_bn --use_reg --m_in -9 --m_out -1 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --use_bn --use_prop --use_reg --m_in -9 --m_out -1 --lamda 0.1 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -9 --m_out -1 --lamda 0.1 --device "$dev"


### Amazon-photo with feature ood

geo train --method msp --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --use_reg --m_in -5 --m_out -1 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --use_prop --use_reg --m_in -9 --m_out -1 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -9 --m_out -1 --lamda 1.0 --device "$dev"


### Amazon-photo with label ood

geo train --method msp --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --use_bn --use_reg --m_in -9 --m_out -4 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --use_bn --use_prop --use_reg --m_in -9 --m_out -4 --lamda 1.0 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -9 --m_out -4 --lamda 1.0 --device "$dev"


### Coauthor with structure ood

geo train --method msp --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -5 --m_out -1 --lamda 0.1 --device "$dev"


### Coauthor with feature ood

geo train --method msp --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --use_reg --m_in -7 --m_out -1 --lamda 0.1 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --use_prop --use_reg --m_in -7 --m_out -1 --lamda 0.1 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -7 --m_out -1 --lamda 0.1 --device "$dev"


### Coauthor with label ood

geo train --method msp --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect \
  --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect \
  --use_bn --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect \
  --use_bn --use_prop --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect \
  --use_bn --use_prop --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect \
  --use_bn --use_prop --use_reg --use_occ --m_in -9 --m_out -2 --lamda 0.01 --device "$dev"


### Arxiv

geo train --method msp --backbone gcn --dataset arxiv --mode detect --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_reg --m_in -9 \
  --m_out -4 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop \
  --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop \
  --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device "$dev"
geo train --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop \
  --use_reg --use_occ --m_in -9 --m_out -2 --lamda 0.01 --device "$dev"
