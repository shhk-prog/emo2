import subprocess
import sys

cmd = [
    "/mnt/nas/home/hiromi/src/emo/.venv/bin/python",
    "scripts/build_all_paper_summaries.py",
    "--strict",
]
print("Running:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
sys.exit(res.returncode)
