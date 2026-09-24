#!/bin/bash
set -e
cd /mnt/nas/home/hiromi/src/emo2
/mnt/nas/home/hiromi/src/emo2/.venv/bin/python3 scripts/summarize_behavioral_emobank.py --out-dir iclr2027/tables
/mnt/nas/home/hiromi/src/emo2/.venv/bin/python3 scripts/summarize_behavioral_aipsy.py --out-dir iclr2027/tables
/mnt/nas/home/hiromi/src/emo2/.venv/bin/python3 scripts/summarize_v1_internal_sharing.py --out-dir iclr2027/tables
/mnt/nas/home/hiromi/src/emo2/.venv/bin/python3 scripts/summarize_v2_reorganization.py --out-dir iclr2027/tables
/mnt/nas/home/hiromi/src/emo2/.venv/bin/python3 scripts/summarize_v3_causal_utilization.py --out-dir iclr2027/tables
echo "ALL TABLES GENERATED SUCCESSFULLY"
