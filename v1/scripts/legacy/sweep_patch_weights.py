#!/usr/bin/env python3
import subprocess

def main():
    run_id = "20260825T023809Z_a1078cb_cb9ad1b8"
    layers = [0, 4, 8, 12, 16, 20, 24, 27]
    weights = [0.5, 0.1]

    for weight in weights:
        print("="*60)
        print(f"Starting Sweep with Patch Weight: {weight}")
        print("="*60)
        for layer in layers:
            print(f"Running patching for layer {layer} with weight {weight}...")
            
            cmd = [
                "python", "scripts/run_causal_intervention.py",
                "--run-id", run_id,
                "--limit", "0",
                "--target-layer", str(layer),
                "--patch-weight", str(weight)
            ]
            
            # サブプロセスとして実行し、出力をそのまま表示
            subprocess.run(cmd, check=False)

    print("All sweeping tasks completed.")

if __name__ == "__main__":
    main()
