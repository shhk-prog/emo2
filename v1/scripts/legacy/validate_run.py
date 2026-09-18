#!/usr/bin/env python3
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Validate the completeness and schema of a run.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to validate.")
    parser.add_argument("--phase", type=str, default=None, help="Experiment phase (e.g., preliminary, main). If None, searches across phases.")
    args = parser.parse_args()

    if args.phase:
        run_dir = os.path.join("results/raw", args.phase, args.run_id)
        if not os.path.isdir(run_dir):
            print(f"Error: Run directory not found: {run_dir}")
            return
    else:
        import glob
        matches = glob.glob(os.path.join("results/raw", "*", args.run_id))
        if not matches:
            print(f"Error: Run directory not found for {args.run_id} in any phase.")
            return
        run_dir = matches[0]
        
    responses_path = os.path.join(run_dir, "responses.jsonl")
    if not os.path.exists(responses_path):
        print(f"Error: {responses_path} not found.")
        return
        
    print(f"Validating {responses_path}...")
    valid_count = 0
    error_count = 0
    
    required_keys = {
        "run_id", "request_id", "baseline_id", "condition", "parse_status"
    }
    
    baseline_ids = set()
    referenced_baseline_ids = set()
    
    with open(responses_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                missing = required_keys - set(data.keys())
                if missing:
                    print(f"Line {line_no}: Missing keys: {missing}")
                    error_count += 1
                else:
                    valid_count += 1
                    cond = data.get("condition")
                    bid = data.get("baseline_id")
                    if cond == "baseline":
                        baseline_ids.add(bid)
                    elif cond in ["recognition", "affective_reception", "empathic_response"]:
                        referenced_baseline_ids.add(bid)
                        
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: Invalid JSON: {e}")
                error_count += 1
                
    print(f"Validation complete. Valid lines: {valid_count}, JSON/Key Errors: {error_count}")
    
    # Check baseline mappings
    missing_baselines = referenced_baseline_ids - baseline_ids
    if missing_baselines:
        print(f"Error: Found references to baseline_ids that do not exist: {missing_baselines}")
        error_count += 1
        
    if error_count == 0:
        print("Run validation passed successfully!")

if __name__ == "__main__":
    main()
