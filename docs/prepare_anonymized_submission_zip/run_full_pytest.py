import os
import subprocess
import sys

env = os.environ.copy()
env["PYTHONPATH"] = "src"

cmd = [
    "/mnt/nas/home/hiromi/src/emo/.venv/bin/python",
    "-m",
    "pytest",
    "-q",
]
print("Running:", " ".join(cmd))
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("Return code:", res.returncode)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
sys.exit(res.returncode)
