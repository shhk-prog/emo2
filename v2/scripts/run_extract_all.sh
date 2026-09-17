#!/bin/bash
set -e
export PYTHONPATH=.

echo "Running extraction for Base model (standard template)..."
.venv/bin/python v2/scripts/run_extract_and_likelihood.py --model Qwen/Qwen2.5-1.5B --template_type standard --limit 0

echo "Running extraction for Instruct model (standard template)..."
.venv/bin/python v2/scripts/run_extract_and_likelihood.py --model Qwen/Qwen2.5-1.5B-Instruct --is_instruct --template_type standard --limit 0

echo "Running extraction for Base model (reversed template)..."
.venv/bin/python v2/scripts/run_extract_and_likelihood.py --model Qwen/Qwen2.5-1.5B --template_type reversed --limit 0

echo "Running extraction for Instruct model (reversed template)..."
.venv/bin/python v2/scripts/run_extract_and_likelihood.py --model Qwen/Qwen2.5-1.5B-Instruct --is_instruct --template_type reversed --limit 0

echo "All extractions completed."
